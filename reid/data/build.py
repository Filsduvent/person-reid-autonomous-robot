from torch.utils.data import DataLoader

from reid.data.collate import test_collate_fn, train_collate_fn
from reid.data.market1501 import Market1501FromPartitions
from reid.data.market1501_test import Market1501TestFromPartitions
from reid.data.samplers import PKBatchSampler
from reid.data.transforms import build_train_tf, build_test_tf

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

    if ds_name != "market1501":
        raise NotImplementedError(f"Step 2.1 supports market1501 only, got {ds_name}")

    dataset = Market1501FromPartitions(root=root, split=split, transform=tf)

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

    if ds_name != "market1501":
        raise NotImplementedError(f"Test loader currently supports market1501 only, got {ds_name}")

    dataset = Market1501TestFromPartitions(root=root, split=split, transform=tf)
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
