from torch.utils.data import DataLoader

from reid.data.collate import test_collate_fn, train_collate_fn
from reid.data.cuhk03 import CUHK03ProcessedTest, CUHK03ProcessedTrain
from reid.data.duke import DukeProcessedTest, DukeProcessedTrain, DukeRawTest, DukeRawTrain
from reid.data.market1501 import Market1501FromPartitions, Market1501RawTrain
from reid.data.market1501_test import Market1501RawTest, Market1501TestFromPartitions
from reid.data.msmt17 import MSMT17RawTest, MSMT17RawTrain
from reid.data.protocol import validate_eval_dataset, validate_train_dataset
from reid.data.samplers import PKBatchSampler
from reid.data.transforms import build_train_tf, build_test_tf


DATASET_DISPLAY_NAMES = {
    "market1501": "Market1501",
    "cuhk03": "CUHK03",
    "duke": "DukeMTMC-ReID",
    "msmt17": "MSMT17",
}


def _camera_count(dataset):
    cams = getattr(dataset, "cams", None)
    if cams is None:
        return "unknown"
    return len({int(cam) for cam in cams if int(cam) >= 0})


def _mark_count(dataset, mark):
    marks = getattr(dataset, "marks", [])
    return sum(1 for value in marks if int(value) == mark)


def _identity_count(dataset, role):
    if role == "train":
        return int(dataset.num_classes)
    pids = getattr(dataset, "pids", None)
    if pids is None:
        return "unknown"
    return len({int(pid) for pid in pids if int(pid) >= 0})


def _query_count(dataset):
    return int(getattr(dataset, "num_query", _mark_count(dataset, 0)))


def _gallery_count(dataset):
    return int(getattr(dataset, "num_gallery", _mark_count(dataset, 1)))


def _print_dataset_summary(dataset_name, dataset_format, dataset, role):
    display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    print(f"[Dataset] name={display_name} role={role}")
    print(f"root: {getattr(dataset, 'root', 'unknown')}")
    print(f"format: {dataset_format}")
    print(f"split: {getattr(dataset, 'split', 'unknown')}")
    if hasattr(dataset, "image_type"):
        print(f"image_type: {dataset.image_type}")
    if hasattr(dataset, "protocol"):
        print(f"protocol: {dataset.protocol}")
    if hasattr(dataset, "split_id"):
        print(f"split_id: {dataset.split_id}")
    print(f"num images: {len(dataset)}")
    print(f"num identities: {_identity_count(dataset, role)}")
    print(f"num cameras: {_camera_count(dataset)}")
    if role == "train":
        print("num query images: n/a")
        print("num gallery images: n/a")
    else:
        print(f"num query images: {_query_count(dataset)}")
        print(f"num gallery images: {_gallery_count(dataset)}")


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


def _build_cuhk03_train_dataset(root, split, dataset_format, dataset_cfg, transform):
    if dataset_format == "processed":
        return CUHK03ProcessedTrain(
            root=root,
            split=split,
            image_type=dataset_cfg.get("image_type", "detected"),
            protocol=dataset_cfg.get("protocol", "new"),
            split_id=dataset_cfg.get("split_id", 0),
            transform=transform,
        )
    raise ValueError(f"Unsupported CUHK03 train dataset format '{dataset_format}'. Use 'processed'.")


def _build_cuhk03_test_dataset(root, split, dataset_format, dataset_cfg, transform):
    if dataset_format == "processed":
        return CUHK03ProcessedTest(
            root=root,
            split=split,
            image_type=dataset_cfg.get("image_type", "detected"),
            protocol=dataset_cfg.get("protocol", "new"),
            split_id=dataset_cfg.get("split_id", 0),
            transform=transform,
        )
    raise ValueError(f"Unsupported CUHK03 test dataset format '{dataset_format}'. Use 'processed'.")


def _build_duke_train_dataset(root, split, dataset_format, transform):
    if dataset_format == "processed":
        return DukeProcessedTrain(root=root, split=split, transform=transform)
    if dataset_format == "raw":
        return DukeRawTrain(root=root, split=split, transform=transform)
    raise ValueError(f"Unsupported Duke train dataset format '{dataset_format}'. Use 'processed' or 'raw'.")


def _build_duke_test_dataset(root, split, dataset_format, transform):
    if dataset_format == "processed":
        return DukeProcessedTest(root=root, split=split, transform=transform)
    if dataset_format == "raw":
        return DukeRawTest(root=root, split=split, transform=transform)
    raise ValueError(f"Unsupported Duke test dataset format '{dataset_format}'. Use 'processed' or 'raw'.")


def _build_msmt17_train_dataset(root, split, dataset_format, transform):
    if dataset_format == "raw":
        return MSMT17RawTrain(root=root, split=split, transform=transform)
    raise ValueError(f"Unsupported MSMT17 train dataset format '{dataset_format}'. Use 'raw'.")


def _build_msmt17_test_dataset(root, split, dataset_format, transform):
    if dataset_format == "raw":
        return MSMT17RawTest(root=root, split=split, transform=transform)
    raise ValueError(f"Unsupported MSMT17 test dataset format '{dataset_format}'. Use 'raw'.")


def build_train_loader(cfg):
    root = cfg["data"]["root"]
    tcfg = cfg["data"]["train"]

    image_size = tuple(tcfg["images"]["size"])
    aug = tcfg["aug"]
    tf = build_train_tf(
        image_size=image_size,
        aug_cfg=aug,
    )

    dataset_cfg = tcfg["dataset"]
    ds_name = dataset_cfg["name"]
    split = dataset_cfg["split"]
    dataset_format = str(dataset_cfg.get("format", "processed")).lower()

    if ds_name == "market1501":
        dataset = _build_market1501_train_dataset(root, split, dataset_format, tf)
    elif ds_name == "cuhk03":
        dataset = _build_cuhk03_train_dataset(root, split, dataset_format, dataset_cfg, tf)
    elif ds_name == "duke":
        dataset = _build_duke_train_dataset(root, split, dataset_format, tf)
    elif ds_name == "msmt17":
        dataset = _build_msmt17_train_dataset(root, split, dataset_format, tf)
    else:
        raise NotImplementedError(f"Train loader supports market1501, cuhk03, duke, and msmt17 only, got {ds_name}")
    validate_train_dataset(dataset)
    _print_dataset_summary(ds_name, dataset_format, dataset, role="train")

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

    loader.effective_batch_size = batch_size
    return loader, int(dataset.num_classes)


def build_test_loader(cfg):
    root = cfg["data"]["root"]
    tcfg = cfg["data"]["test"]

    image_size = tuple(tcfg["images"]["size"])
    aug = tcfg["aug"]
    tf = build_test_tf(image_size=image_size, mean=aug["mean"], std=aug["std"])

    dataset_cfg = tcfg["dataset"]
    ds_name = dataset_cfg["name"]
    split = dataset_cfg["split"]
    dataset_format = str(dataset_cfg.get("format", "processed")).lower()

    if ds_name == "market1501":
        dataset = _build_market1501_test_dataset(root, split, dataset_format, tf)
    elif ds_name == "cuhk03":
        dataset = _build_cuhk03_test_dataset(root, split, dataset_format, dataset_cfg, tf)
    elif ds_name == "duke":
        dataset = _build_duke_test_dataset(root, split, dataset_format, tf)
    elif ds_name == "msmt17":
        dataset = _build_msmt17_test_dataset(root, split, dataset_format, tf)
    else:
        raise NotImplementedError(f"Test loader supports market1501, cuhk03, duke, and msmt17 only, got {ds_name}")
    validate_eval_dataset(dataset)
    _print_dataset_summary(ds_name, dataset_format, dataset, role="test")
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
