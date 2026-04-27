import copy

import pytest
import torch

from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim import (
    WarmupMultiStepLR,
    build_center_optimizer,
    build_optimizer,
    build_scheduler,
)


BASE_CFG = {
    "model": {
        "name": "reid_baseline",
        "backbone": {
            "name": "resnet50",
            "pretrained": False,
            "last_conv_stride": 1,
        },
        "head": {
            "embedding_dim": 32,
            "pooling": "gap",
            "bnneck": True,
            "normalize": True,
            "metric_feat": "bn",
            "classifier": True,
        },
    },
    "loss": {
        "triplet": {
            "enabled": True,
            "margin": 0.3,
            "mining": "batch_hard",
            "weight": 1.0,
        },
        "id": {
            "enabled": True,
            "label_smoothing": 0.1,
            "weight": 1.0,
        },
        "center": {
            "enabled": False,
            "weight": 0.0005,
            "lr": 0.5,
            "feat": "raw",
        },
    },
    "optim": {
        "name": "adam",
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "bias_lr_factor": 2.0,
        "weight_decay_bias": 1e-4,
        "momentum": 0.9,
    },
    "sched": {
        "name": "warmup_multistep",
        "milestones": [2, 4],
        "gamma": 0.1,
        "warmup_factor": 0.5,
        "warmup_iters": 2,
        "warmup_method": "linear",
    },
}


def _make_cfg():
    return copy.deepcopy(BASE_CFG)


@pytest.mark.parametrize(
    ("optim_name", "expected_type"),
    [
        ("sgd", torch.optim.SGD),
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
    ],
)
def test_build_optimizer_supports_all_expected_types(optim_name, expected_type):
    cfg = _make_cfg()
    cfg["optim"]["name"] = optim_name
    model = build_model(cfg, num_classes=3)

    optimizer = build_optimizer(cfg, model)

    assert isinstance(optimizer, expected_type)
    assert optimizer.param_groups


def test_build_optimizer_applies_bias_group_overrides():
    cfg = _make_cfg()
    model = build_model(cfg, num_classes=3)

    optimizer = build_optimizer(cfg, model)

    base_lr = float(cfg["optim"]["lr"])
    bias_lr = base_lr * float(cfg["optim"]["bias_lr_factor"])
    base_wd = float(cfg["optim"]["weight_decay"])
    bias_wd = float(cfg["optim"]["weight_decay_bias"])

    observed_lrs = {group["lr"] for group in optimizer.param_groups}
    observed_wds = {group["weight_decay"] for group in optimizer.param_groups}

    assert base_lr in observed_lrs
    assert bias_lr in observed_lrs
    assert base_wd in observed_wds
    assert bias_wd in observed_wds


def test_build_center_optimizer_only_when_enabled():
    cfg = _make_cfg()
    model = build_model(cfg, num_classes=3)
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)

    assert build_center_optimizer(cfg, criterion) is None

    cfg["loss"]["center"]["enabled"] = True
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)
    center_optimizer = build_center_optimizer(cfg, criterion)

    assert isinstance(center_optimizer, torch.optim.SGD)
    assert center_optimizer.param_groups[0]["lr"] == pytest.approx(cfg["loss"]["center"]["lr"])


def test_build_scheduler_supports_warmup_multistep_and_updates_per_iteration():
    cfg = _make_cfg()
    cfg["sched"]["milestones"] = [4, 6]
    model = build_model(cfg, num_classes=3)
    optimizer = build_optimizer(cfg, model)

    scheduler = build_scheduler(cfg, optimizer)

    assert isinstance(scheduler, WarmupMultiStepLR)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < lrs[1] < lrs[2]
    assert lrs[2] == pytest.approx(cfg["optim"]["lr"])
    assert lrs[4] < lrs[3]


def test_build_scheduler_supports_legacy_step_config():
    cfg = _make_cfg()
    cfg["sched"] = {
        "name": "step",
        "milestones": [2, 4],
        "gamma": 0.1,
    }
    model = build_model(cfg, num_classes=3)
    optimizer = build_optimizer(cfg, model)

    scheduler = build_scheduler(cfg, optimizer)

    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
