from torch.utils.data import DataLoader

from reid.data.market1501 import Market1501FromPartitions
from reid.data.samplers import PKBatchSampler
from reid.data.transforms import build_train_tf, build_test_tf

def build_train_loader(cfg):
    root = cfg["data"]["root"]
    tcfg = cfg["data"]["train"]

    image_size = tuple(tcfg["images"]["size"])
    aug = tcfg["aug"]
    tf = build_train_tf(
        image_size=image_size,
        mean=aug["mean"],
        std=aug["std"],
        mirror=aug["mirror"],
        crop_prob=aug["crop_prob"],
        crop_ratio=aug["crop_ratio"],
        scale_255=aug["scale_255"],
    )

    ds_name = tcfg["dataset"]["name"]
    split = tcfg["dataset"]["split"]

    if ds_name != "market1501":
        raise NotImplementedError(f"Step 2.1 supports market1501 only, got {ds_name}")

    dataset = Market1501FromPartitions(root=root, split=split, transform=tf)

    batch_cfg = tcfg["batch"]
    sampler_name = batch_cfg["sampler"]
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    if sampler_name == "pk":
        P = int(batch_cfg["P"])
        K = int(batch_cfg["K"])
        batch_sampler = PKBatchSampler(dataset.labels, P=P, K=K, drop_last=True, seed=cfg["repro"]["seed"])
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        steps_per_epoch = None  # we will control this in train loop via cfg if needed later
        batch_size = P * K
    else:
        raise NotImplementedError("Only pk sampler is supported in Step 2.1")

    return loader, batch_size
