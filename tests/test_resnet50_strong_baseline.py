import pytest
import torch

from reid.data.transforms import RandomErasing, build_test_tf, build_train_tf
from reid.losses.build import build_criterion
from reid.losses.center import CenterLoss
from reid.losses.id import CrossEntropyLabelSmooth
from reid.losses.triplet import BatchHardTripletLoss
from reid.models.build import build_model
from reid.optim import WarmupMultiStepLR, build_center_optimizer, build_optimizer, build_scheduler
from reid.utils.config import load_config


BASELINE_CONFIG = "configs/baseline_market1501_resnet50_triplet.yaml"


def _baseline_cfg():
    cfg = load_config(BASELINE_CONFIG)
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["head"]["embedding_dim"] = 32
    return cfg


def _mode_cfg(mode):
    cfg = _baseline_cfg()
    cfg["loss"]["triplet"]["enabled"] = mode in {"triplet", "triplet_id", "triplet_id_center"}
    cfg["loss"]["id"]["enabled"] = mode in {"id", "triplet_id", "triplet_id_center"}
    cfg["loss"]["center"]["enabled"] = mode == "triplet_id_center"
    cfg["model"]["head"]["classifier"] = cfg["loss"]["id"]["enabled"] or cfg["loss"]["center"]["enabled"]
    cfg["data"]["train"]["batch"]["sampler"] = "pk" if cfg["loss"]["triplet"]["enabled"] else "random"
    cfg["data"]["train"]["batch"]["batch_size"] = 4
    cfg["data"]["train"]["batch"]["P"] = 2
    cfg["data"]["train"]["batch"]["K"] = 2
    return cfg


def test_resnet50_strong_baseline_config_declares_expected_tricks():
    cfg = load_config(BASELINE_CONFIG)
    train_aug = cfg["data"]["train"]["aug"]
    test_aug = cfg["data"]["test"]["aug"]

    assert cfg["model"]["name"] == "reid_baseline"
    assert cfg["model"]["backbone"]["name"] == "resnet50"
    assert cfg["model"]["backbone"]["last_conv_stride"] in {1, 2}
    assert cfg["model"]["head"]["bnneck"] is True
    assert cfg["model"]["head"]["metric_feat"] in {"raw", "bn"}
    assert cfg["model"]["head"]["eval_feat"] in {"raw", "bn"}

    assert cfg["loss"]["triplet"]["enabled"] is True
    assert cfg["loss"]["id"]["enabled"] is True
    assert cfg["loss"]["id"]["label_smoothing"] > 0.0
    assert "enabled" in cfg["loss"]["center"]
    assert "lr" in cfg["loss"]["center"]

    assert train_aug["random_erasing"]["enabled"] is True
    train_tf = build_train_tf(tuple(cfg["data"]["train"]["images"]["size"]), train_aug)
    test_tf = build_test_tf(tuple(cfg["data"]["test"]["images"]["size"]), test_aug["mean"], test_aug["std"])
    assert any(isinstance(op, RandomErasing) for op in train_tf.transforms)
    assert not any(isinstance(op, RandomErasing) for op in test_tf.transforms)

    assert cfg["sched"]["name"] == "warmup_multistep"
    assert cfg["optim"]["bias_lr_factor"] >= 1.0
    assert cfg["data"]["train"]["batch"]["sampler"] == "pk"
    assert cfg["data"]["train"]["batch"]["P"] > 1
    assert cfg["data"]["train"]["batch"]["K"] > 1
    assert "rerank" in cfg["eval"]
    assert {"enabled", "k1", "k2", "lambda_value"}.issubset(cfg["eval"]["rerank"].keys())


@pytest.mark.parametrize(
    "mode",
    ["triplet", "id", "triplet_id", "triplet_id_center"],
)
def test_resnet50_strong_baseline_loss_modes_are_yaml_only_smokes(mode):
    cfg = _mode_cfg(mode)
    num_classes = 3 if cfg["model"]["head"]["classifier"] else None
    model = build_model(cfg, num_classes=num_classes)
    criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=model.feat_dim)
    optimizer = build_optimizer(cfg, model)
    center_optimizer = build_center_optimizer(cfg, criterion)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=1)

    assert isinstance(scheduler, WarmupMultiStepLR)
    assert (criterion.triplet is not None) is cfg["loss"]["triplet"]["enabled"]
    assert (criterion.id_loss is not None) is cfg["loss"]["id"]["enabled"]
    assert (criterion.center_loss is not None) is cfg["loss"]["center"]["enabled"]
    assert (center_optimizer is not None) is cfg["loss"]["center"]["enabled"]
    if criterion.triplet is not None:
        assert isinstance(criterion.triplet, BatchHardTripletLoss)
    if criterion.id_loss is not None:
        assert isinstance(criterion.id_loss, CrossEntropyLabelSmooth)
    if criterion.center_loss is not None:
        assert isinstance(criterion.center_loss, CenterLoss)

    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    outputs = model(torch.randn(4, 3, 64, 32))
    loss, logs = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if center_optimizer is not None:
        center_optimizer.step()
    scheduler.step()

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    if cfg["loss"]["id"]["enabled"] or cfg["loss"]["center"]["enabled"]:
        assert logs["loss/total"] > 0.0
    else:
        assert logs["loss/total"] >= 0.0
    assert set(outputs) >= {"feat_raw", "feat_bn", "emb", "logits"}
