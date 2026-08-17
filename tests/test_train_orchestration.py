import copy
import json
import logging
from pathlib import Path

import pytest
import torch

from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim import build_center_optimizer, build_optimizer, build_scheduler
from reid.utils.checkpoint import save_checkpoint
from reid.utils.logger import setup_logger
from scripts.train import (
    get_checkpoint_paths,
    is_better_score,
    maybe_resume_training,
    save_eval_metrics,
    save_resolved_config,
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
    "train": {
        "save": {
            "metric": "mAP",
            "resume": "",
        }
    },
}


def _make_cfg():
    return copy.deepcopy(BASE_CFG)


def test_save_resolved_config_writes_canonical_and_compatibility_files(tmp_path):
    cfg = _make_cfg()

    resolved_path = save_resolved_config(str(tmp_path), cfg)

    assert resolved_path.endswith("config.resolved.yaml")
    assert (tmp_path / "config.resolved.yaml").exists()
    assert (tmp_path / "config.yaml").exists()


def test_get_checkpoint_paths_uses_ckpt_names(tmp_path):
    paths = get_checkpoint_paths(str(tmp_path))

    assert paths["best"].endswith("ckpt_best.pth")
    assert paths["last"].endswith("ckpt_last.pth")


def test_train_logger_writes_train_log(tmp_path):
    logger = setup_logger("test.train.log.artifact", str(tmp_path), filename="train.log")

    logger.info("epoch=1 loss_total=1.0 lr=0.001 speed=10 imgs/s")

    for handler in logger.handlers:
        handler.flush()
    log_text = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert "epoch=1" in log_text
    assert "loss_total=1.0" in log_text
    assert "lr=0.001" in log_text
    assert "speed=10 imgs/s" in log_text


def test_save_eval_metrics_writes_latest_and_epoch_test_files(tmp_path):
    cfg = {
        "model": {
            "name": "reid_baseline",
            "backbone": {
                "name": "resnet50",
            },
        },
        "data": {
            "test": {
                "dataset": {
                    "name": "market1501",
                    "split": "test",
                }
            }
        }
    }
    scores = {
        "mAP": 0.5,
        "mINP": 0.4,
        "Rank1": 0.7,
        "Rank5": 0.8,
        "Rank10": 0.9,
    }

    paths = save_eval_metrics(str(tmp_path), cfg, epoch=3, scores=scores, checkpoint_name="ckpt_last.pth")

    assert paths["latest"].endswith("latest_val.json")
    assert paths["epoch"].endswith("val_epoch_003.json")
    latest = json.loads((tmp_path / "latest_val.json").read_text())
    epoch = json.loads((tmp_path / "val_epoch_003.json").read_text())
    assert latest == epoch
    assert latest["dataset"] == "market1501"
    assert latest["split"] == "test"
    assert latest["model"] == "resnet50_baseline"
    assert latest["epoch"] == 3
    assert latest["checkpoint"] == "ckpt_last.pth"
    assert latest["mAP"] == 0.5


def test_is_better_score_uses_configured_metric():
    assert is_better_score({"mAP": 0.6, "Rank1": 0.8}, "mAP", 0.5) is True
    assert is_better_score({"mAP": 0.4, "Rank1": 0.8}, "mAP", 0.5) is False

    with pytest.raises(KeyError, match="Configured checkpoint metric"):
        is_better_score({"Rank1": 0.8}, "mAP", 0.5)


def test_maybe_resume_training_restores_epoch_metric_and_full_state(tmp_path):
    cfg = _make_cfg()
    logger = logging.getLogger("test_train_resume")
    model = build_model(cfg, num_classes=3)
    criterion = build_criterion(cfg, num_classes=3, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=2)
    center_optimizer = build_center_optimizer(cfg, criterion)
    ckpt_path = Path(tmp_path) / "checkpoints" / "ckpt_last.pth"

    optimizer.step()
    scheduler.step()
    if center_optimizer is not None:
        center_optimizer.step()

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        center_optimizer=center_optimizer,
        epoch=4,
        scores={"mAP": 0.88},
        cfg=cfg,
    )

    cfg["train"]["save"]["resume"] = str(ckpt_path)

    model2 = build_model(cfg, num_classes=3)
    criterion2 = build_criterion(cfg, num_classes=3, feat_dim=model2.feat_dim)
    optimizer2 = build_optimizer(cfg, model2)
    scheduler2 = build_scheduler(cfg, optimizer2, steps_per_epoch=2)
    center_optimizer2 = build_center_optimizer(cfg, criterion2)

    start_epoch, best_metric, checkpoint = maybe_resume_training(
        logger=logger,
        cfg=cfg,
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        center_optimizer=center_optimizer2,
        device="cpu",
    )

    assert start_epoch == 5
    assert best_metric == pytest.approx(0.88)
    assert checkpoint["epoch"] == 4
    assert optimizer2.state_dict() == optimizer.state_dict()
    assert scheduler2.state_dict() == scheduler.state_dict()
    assert center_optimizer2.state_dict() == center_optimizer.state_dict()
