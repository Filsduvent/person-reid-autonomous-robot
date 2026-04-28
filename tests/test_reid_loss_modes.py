import copy

import pytest
import torch
import torch.nn.functional as F

from reid.losses.build import build_criterion
from reid.losses.id import CrossEntropyLabelSmooth, IDCrossEntropyLoss, build_id_loss
from reid.models.build import build_model
from reid.models.outputs import ensure_output_dict


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
            "enabled": False,
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
            "feat": "raw",
        },
    },
}


def _make_cfg(*, triplet: bool, id_loss: bool, center: bool):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["loss"]["triplet"]["enabled"] = triplet
    cfg["loss"]["id"]["enabled"] = id_loss
    cfg["loss"]["center"]["enabled"] = center
    cfg["model"]["head"]["classifier"] = id_loss or center
    return cfg


@pytest.mark.parametrize(
    ("mode_name", "cfg"),
    [
        ("triplet_only", _make_cfg(triplet=True, id_loss=False, center=False)),
        ("id_only", _make_cfg(triplet=False, id_loss=True, center=False)),
        ("triplet_id", _make_cfg(triplet=True, id_loss=True, center=False)),
        ("triplet_id_center", _make_cfg(triplet=True, id_loss=True, center=True)),
    ],
)
def test_reid_modes_one_batch_cpu(mode_name, cfg):
    del mode_name  # parameter name is useful in pytest output, not in the body

    batch_size = 4
    num_classes = 3 if cfg["model"]["head"]["classifier"] else None
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    imgs = torch.randn(batch_size, 3, 256, 128, dtype=torch.float32)

    model = build_model(cfg, num_classes=num_classes)
    criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=model.feat_dim)

    outputs = ensure_output_dict(model(imgs))
    total_loss, logs = criterion(outputs, labels)

    assert isinstance(outputs, dict)
    assert outputs["emb"].shape[0] == batch_size
    assert torch.is_tensor(total_loss)
    assert total_loss.ndim == 0

    expected_log_keys = {"loss/total", "loss/triplet", "loss/id", "loss/center"}
    assert expected_log_keys.issubset(logs.keys())

    if cfg["model"]["head"]["classifier"]:
        assert outputs["logits"] is not None
        assert outputs["logits"].shape == (batch_size, num_classes)
    else:
        assert outputs["logits"] is None

    total_loss.backward()

    if cfg["loss"]["center"]["enabled"]:
        assert criterion.center is not None
        assert criterion.center.centers.grad is not None
        assert criterion.center.centers.grad.shape == criterion.center.centers.shape
    else:
        assert criterion.center is None


def test_build_id_loss_switches_between_ce_and_label_smoothing():
    assert isinstance(build_id_loss(label_smoothing=0.0), IDCrossEntropyLoss)
    assert isinstance(build_id_loss(label_smoothing=0.1), CrossEntropyLabelSmooth)


def test_label_smoothing_zero_matches_cross_entropy():
    logits = torch.tensor(
        [[2.5, 0.1, -1.3], [0.2, 1.7, -0.4], [-0.8, 0.3, 2.2]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    ce_loss = F.cross_entropy(logits, targets)
    smooth_zero_loss = build_id_loss(label_smoothing=0.0)(logits, targets)

    assert smooth_zero_loss == pytest.approx(float(ce_loss), rel=1e-6, abs=1e-7)


def test_label_smoothing_positive_changes_loss_value():
    logits = torch.tensor(
        [[3.0, 0.5, -1.0], [0.1, 2.0, -0.5], [-1.2, 0.3, 2.5]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    ce_loss = build_id_loss(label_smoothing=0.0)(logits, targets)
    smooth_loss = build_id_loss(label_smoothing=0.1)(logits, targets)

    assert float(smooth_loss) != pytest.approx(float(ce_loss), rel=1e-6, abs=1e-7)
