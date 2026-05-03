from pathlib import Path

import pytest
import torch

from reid.models.build import build_model
from reid.optim.build import build_optimizer, build_scheduler
from reid.losses.build import build_criterion
from reid.optim import build_center_optimizer
from reid.utils.checkpoint import load_checkpoint, save_checkpoint
from scripts.evaluate import infer_num_classes_from_checkpoint


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
            "enabled": True,
            "weight": 0.0005,
            "lr": 0.5,
        },
    },
    "optim": {
        "name": "adam",
        "lr": 3e-4,
        "weight_decay": 5e-4,
        "bias_lr_factor": 1.0,
        "weight_decay_bias": 5e-4,
        "momentum": 0.9,
    },
    "sched": {
        "name": "warmup_multistep",
        "milestones": [2, 3],
        "gamma": 0.1,
        "warmup_factor": 0.5,
        "warmup_iters": 2,
        "warmup_method": "linear",
    },
}


def test_save_checkpoint_writes_complete_training_state(tmp_path):
    cfg = BASE_CFG
    model = build_model(cfg, num_classes=3)
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=2)
    center_optimizer = build_center_optimizer(cfg, criterion)
    path = tmp_path / "checkpoints" / "last.pth"

    payload = save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        center_optimizer=center_optimizer,
        epoch=7,
        scores={"mAP": 0.9},
        cfg=cfg,
    )

    assert path.exists()
    assert set(payload) == {
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "center_optimizer",
        "scores",
        "cfg",
    }
    assert payload["epoch"] == 7
    assert payload["model"].keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(payload["model"][key], value)
    assert payload["optimizer"] is not None
    assert payload["scheduler"] is not None
    assert payload["center_optimizer"] is not None
    assert payload["scores"] == {"mAP": 0.9}
    assert payload["cfg"] == cfg


def test_load_checkpoint_restores_optimizer_scheduler_and_center_state(tmp_path):
    cfg = BASE_CFG
    model = build_model(cfg, num_classes=3)
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=2)
    center_optimizer = build_center_optimizer(cfg, criterion)
    path = Path(tmp_path) / "last.pth"

    optimizer.step()
    scheduler.step()
    if center_optimizer is not None:
        center_optimizer.step()

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        center_optimizer=center_optimizer,
        epoch=5,
        scores={"mAP": 0.8},
        cfg=cfg,
    )

    model2 = build_model(cfg, num_classes=3)
    criterion2 = build_criterion(cfg, num_classes=3, feat_dim=model2.feat_dim)
    optimizer2 = build_optimizer(cfg, model2)
    scheduler2 = build_scheduler(cfg, optimizer2, steps_per_epoch=2)
    center_optimizer2 = build_center_optimizer(cfg, criterion2)

    checkpoint = load_checkpoint(
        path=path,
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        center_optimizer=center_optimizer2,
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 5
    assert checkpoint["scores"] == {"mAP": 0.8}
    assert optimizer2.state_dict() == optimizer.state_dict()
    assert scheduler2.state_dict() == scheduler.state_dict()
    assert center_optimizer2.state_dict() == center_optimizer.state_dict()


def test_load_checkpoint_without_optional_optimizers_still_restores_model(tmp_path):
    cfg = BASE_CFG
    cfg = {
        **cfg,
        "loss": {
            **cfg["loss"],
            "center": {
                **cfg["loss"]["center"],
                "enabled": False,
            },
        },
    }
    model = build_model(cfg, num_classes=3)
    path = Path(tmp_path) / "model_only.pth"

    save_checkpoint(path=path, model=model, epoch=3, scores={"mAP": 0.7}, cfg=cfg)

    model2 = build_model(cfg, num_classes=3)
    checkpoint = load_checkpoint(path=path, model=model2, map_location="cpu")

    assert checkpoint["epoch"] == 3
    for key, value in model.state_dict().items():
        assert torch.equal(value, model2.state_dict()[key])


def test_load_checkpoint_supports_raw_model_state_dict(tmp_path):
    cfg = BASE_CFG
    model = build_model(cfg, num_classes=3)
    path = Path(tmp_path) / "model_only_state_dict.pth"
    torch.save(model.state_dict(), path)

    model2 = build_model(cfg, num_classes=3)
    checkpoint = load_checkpoint(path=path, model=model2, map_location="cpu")

    assert checkpoint.keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, model2.state_dict()[key])


def test_load_checkpoint_strips_dataparallel_module_prefix(tmp_path):
    cfg = BASE_CFG
    model = build_model(cfg, num_classes=3)
    module_state = {f"module.{key}": value.clone() for key, value in model.state_dict().items()}
    path = Path(tmp_path) / "module_prefixed_state_dict.pth"
    torch.save(module_state, path)

    model2 = build_model(cfg, num_classes=3)
    load_checkpoint(path=path, model=model2, map_location="cpu")

    for key, value in model.state_dict().items():
        assert torch.equal(value, model2.state_dict()[key])


def test_standalone_evaluation_can_infer_classes_and_load_best_checkpoint(tmp_path):
    cfg = BASE_CFG
    model = build_model(cfg, num_classes=3)
    path = Path(tmp_path) / "checkpoints" / "ckpt_best.pth"

    checkpoint = save_checkpoint(
        path=path,
        model=model,
        epoch=9,
        scores={"mAP": 0.91},
        cfg=cfg,
        is_best=True,
    )

    assert infer_num_classes_from_checkpoint(checkpoint) == 3

    model2 = build_model(cfg, num_classes=3)
    loaded = load_checkpoint(path=path, model=model2, map_location="cpu")

    assert loaded["epoch"] == 9
    assert loaded["scores"] == {"mAP": 0.91}
    for key, value in model.state_dict().items():
        assert torch.equal(value, model2.state_dict()[key])
