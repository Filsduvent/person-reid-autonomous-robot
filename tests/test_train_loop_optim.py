import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from reid.engine.train_loop import train_one_epoch
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim import build_center_optimizer, build_optimizer, build_scheduler


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
            "classifier": False,
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
            "enabled": False,
            "label_smoothing": 0.1,
            "weight": 1.0,
        },
        "center": {
            "enabled": False,
            "weight": 0.0005,
            "lr": 0.5,
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
        "milestones": [4, 6],
        "gamma": 0.1,
        "warmup_factor": 0.5,
        "warmup_iters": 2,
        "warmup_method": "linear",
    },
}


def _make_cfg(*, triplet=True, id_loss=False, center=False):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["loss"]["triplet"]["enabled"] = triplet
    cfg["loss"]["id"]["enabled"] = id_loss
    cfg["loss"]["center"]["enabled"] = center
    cfg["model"]["head"]["classifier"] = id_loss or center
    return cfg


@pytest.mark.parametrize(
    ("cfg", "expect_center_optimizer"),
    [
        (_make_cfg(triplet=True, id_loss=False, center=False), False),
        (_make_cfg(triplet=False, id_loss=True, center=False), False),
        (_make_cfg(triplet=True, id_loss=True, center=False), False),
        (_make_cfg(triplet=True, id_loss=True, center=True), True),
    ],
)
def test_train_loop_runs_with_expected_optimizers_and_logs_lr(cfg, expect_center_optimizer, capsys):
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    imgs = torch.randn(4, 3, 64, 32, dtype=torch.float32)
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=4, shuffle=False)

    num_classes = 3 if cfg["model"]["head"]["classifier"] else None
    model = build_model(cfg, num_classes=num_classes)
    criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    center_optimizer = build_center_optimizer(cfg, criterion)
    scheduler = build_scheduler(cfg, optimizer)

    assert (center_optimizer is not None) is expect_center_optimizer

    avg_loss = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        center_optimizer=center_optimizer,
        device=torch.device("cpu"),
        amp=False,
        log_interval=1,
        scheduler=scheduler,
        tb_writer=None,
        epoch=1,
        steps_per_epoch=1,
    )

    captured = capsys.readouterr()
    assert avg_loss > 0.0
    assert " lr=" in captured.out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this environment")
@pytest.mark.parametrize(
    "cfg",
    [
        _make_cfg(triplet=False, id_loss=True, center=False),
        _make_cfg(triplet=True, id_loss=True, center=False),
        _make_cfg(triplet=True, id_loss=True, center=True),
    ],
)
def test_train_loop_runs_on_cuda_for_loss_combinations(cfg):
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    imgs = torch.randn(4, 3, 64, 32, dtype=torch.float32)
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=4, shuffle=False)

    num_classes = 3 if cfg["model"]["head"]["classifier"] else None
    model = build_model(cfg, num_classes=num_classes).to("cuda")
    criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=model.feat_dim).to("cuda")
    optimizer = build_optimizer(cfg, model)
    center_optimizer = build_center_optimizer(cfg, criterion)
    scheduler = build_scheduler(cfg, optimizer)

    avg_loss = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        center_optimizer=center_optimizer,
        device=torch.device("cuda"),
        amp=False,
        log_interval=1,
        scheduler=scheduler,
        tb_writer=None,
        epoch=1,
        steps_per_epoch=1,
    )

    assert avg_loss > 0.0
