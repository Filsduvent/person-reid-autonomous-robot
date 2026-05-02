from torch.utils.data import DataLoader

from reid.data.collate import test_collate_fn, train_collate_fn
from reid.data.market1501 import Market1501FromPartitions, Market1501RawTrain
from reid.data.market1501_test import Market1501RawTest, Market1501TestFromPartitions
from reid.data.protocol import validate_eval_dataset, validate_train_dataset
from reid.data.samplers import PKBatchSampler
from reid.data.transforms import build_train_tf, build_test_tf


def _camera_count(dataset):
    cams = getattr(dataset, "cams", None)
    if cams is None:
        return "unknown"
    return len({int(cam) for cam in cams if int(cam) >= 0})


def _mark_count(dataset, mark):
    marks = getattr(dataset, "marks", [])
    return sum(1 for value in marks if int(value) == mark)


def _build_market1501_train_dataset(root, split, dataset_format, transform):
    if dataset_format == "processed":
        return Market1501FromPartitions(root=root, split=split, transform=transform)
    if dataset_format == "raw":
        return Market1501RawTrain(root=root, split=split, transform=transform)
    raise ValueError(f"Unsupported Market1501 train dataset format '{dataset_format}'. Use 'processed' or 'raw'.")


def _build_market1501_test_dataset(root, split, dataset_format, transform):
    if dataset_format == "processed":
        return Market1501TestFromPartitions(root=root, split=split, transform=transform)
    if dataset_format == "raw":
        return Market1501RawTest(root=root, split=split, transform=transform)
    raise ValueError(f"Unsupported Market1501 test dataset format '{dataset_format}'. Use 'processed' or 'raw'.")


def _print_market1501_train_stats(dataset_format, dataset):
    print(f"[Market1501] format={dataset_format}")
    print(f"train identities: {int(dataset.num_classes)}")
    print(f"train images: {len(dataset)}")
    print("query images: n/a")
    print("gallery images: n/a")
    print(f"cameras: {_camera_count(dataset)}")


def _print_market1501_test_stats(dataset_format, dataset):
    print(f"[Market1501] format={dataset_format}")
    print("train identities: n/a")
    print("train images: n/a")
    print(f"query images: {_mark_count(dataset, 0)}")
    print(f"gallery images: {_mark_count(dataset, 1)}")
    print(f"cameras: {_camera_count(dataset)}")


def build_train_loader(cfg):
    root = cfg["data"]["root"]
    tcfg = cfg["data"]["train"]

    image_size = tuple(tcfg["images"]["size"])
    aug = tcfg["aug"]
    tf = build_train_tf(
        image_size=image_size,
        aug_cfg=aug,
    )

    ds_name = tcfg["dataset"]["name"]
    split = tcfg["dataset"]["split"]
    dataset_format = str(tcfg["dataset"].get("format", "processed")).lower()

    if ds_name != "market1501":
        raise NotImplementedError(f"Step 2.1 supports market1501 only, got {ds_name}")

    dataset = _build_market1501_train_dataset(root, split, dataset_format, tf)
    validate_train_dataset(dataset)
    _print_market1501_train_stats(dataset_format, dataset)

    batch_cfg = tcfg["batch"]
    sampler_name = str(batch_cfg.get("sampler", "pk")).lower()
    triplet_enabled = bool(cfg["loss"]["triplet"]["enabled"])
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    if triplet_enabled and sampler_name != "pk":
        raise ValueError("Triplet loss requires sampler='pk' so each batch has positive pairs.")

    if sampler_name == "pk":
        P = int(batch_cfg["P"])
        K = int(batch_cfg["K"])
        if P <= 1:
            raise ValueError("PK sampler requires P > 1 so each batch has multiple identities.")
        if K <= 1:
            raise ValueError("PK sampler requires K > 1 so each identity has positive pairs.")
        expected_batch_size = P * K
        configured_batch_size = int(batch_cfg.get("batch_size", expected_batch_size))
        if configured_batch_size != expected_batch_size:
            raise ValueError(
                f"PK sampler requires batch_size == P*K, got batch_size={configured_batch_size} "
                f"and P*K={expected_batch_size}."
            )
        batch_sampler = PKBatchSampler(dataset.labels, P=P, K=K, drop_last=True, seed=cfg["repro"]["seed"])
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=train_collate_fn,
        )
        batch_size = expected_batch_size
    elif sampler_name == "random":
        batch_size = int(batch_cfg["batch_size"])
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=train_collate_fn,
        )
    else:
        raise ValueError(f"Unsupported train sampler '{sampler_name}'. Use 'pk' or 'random'.")

    return loader, batch_size


def build_test_loader(cfg):
    root = cfg["data"]["root"]
    tcfg = cfg["data"]["test"]

    image_size = tuple(tcfg["images"]["size"])
    aug = tcfg["aug"]
    tf = build_test_tf(image_size=image_size, mean=aug["mean"], std=aug["std"])

    ds_name = tcfg["dataset"]["name"]
    split = tcfg["dataset"]["split"]
    dataset_format = str(tcfg["dataset"].get("format", "processed")).lower()

    if ds_name != "market1501":
        raise NotImplementedError(f"Test loader currently supports market1501 only, got {ds_name}")

    dataset = _build_market1501_test_dataset(root, split, dataset_format, tf)
    validate_eval_dataset(dataset)
    _print_market1501_test_stats(dataset_format, dataset)
    loader = DataLoader(
        dataset,
        batch_size=int(tcfg["batch"]["size"]),
        shuffle=bool(tcfg["loader"]["shuffle"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
        collate_fn=test_collate_fn,
    )
    return loader
