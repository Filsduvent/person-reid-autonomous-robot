import copy

import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

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
        assert criterion.center_loss is not None
        assert criterion.center is criterion.center_loss
        assert criterion.center_loss.centers.grad is not None
        assert criterion.center_loss.centers.grad.shape == criterion.center_loss.centers.shape
    else:
        assert criterion.center_loss is None
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


class _RecordTriplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_x = None
        self.last_labels = None

    def forward(self, x, labels):
        self.last_x = x.detach().clone()
        self.last_labels = labels.detach().clone()
        return x.sum() * 0.0


class _RecordCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_x = None
        self.last_labels = None

    def forward(self, x, labels):
        self.last_x = x.detach().clone()
        self.last_labels = labels.detach().clone()
        return x.sum() * 0.0


class _RecordID(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_logits = None
        self.last_labels = None

    def forward(self, logits, labels):
        self.last_logits = logits.detach().clone()
        self.last_labels = labels.detach().clone()
        return logits.sum() * 0.0


@pytest.mark.parametrize(
    ("metric_feat", "expected_key"),
    [
        ("raw", "feat_raw"),
        ("bn", "feat_bn"),
    ],
)
def test_metric_losses_use_configured_metric_feature(metric_feat, expected_key):
    cfg = _make_cfg(triplet=True, id_loss=False, center=True)
    cfg["model"]["head"]["metric_feat"] = metric_feat
    cfg["loss"]["triplet"]["weight"] = 0.0
    cfg["loss"]["center"]["weight"] = 0.0

    criterion = build_criterion(cfg, num_classes=3, feat_dim=32)
    triplet = _RecordTriplet()
    center = _RecordCenter()
    criterion.triplet = triplet
    criterion.center_loss = center
    criterion.center = center

    outputs = {
        "feat_raw": torch.randn(4, 32),
        "feat_bn": torch.randn(4, 32),
        "emb": torch.randn(4, 32),
        "logits": torch.randn(4, 3),
    }
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    total_loss, logs = criterion(outputs, labels)

    assert torch.is_tensor(total_loss)
    assert "loss/triplet" in logs
    assert "loss/center" in logs
    assert torch.allclose(triplet.last_x, outputs[expected_key])
    assert torch.allclose(center.last_x, outputs[expected_key])
    assert torch.equal(triplet.last_labels, labels)
    assert torch.equal(center.last_labels, labels)


def test_id_loss_uses_logits_only():
    cfg = _make_cfg(triplet=False, id_loss=True, center=False)
    criterion = build_criterion(cfg, num_classes=3, feat_dim=32)
    id_loss = _RecordID()
    criterion.id_loss = id_loss

    outputs = {
        "feat_raw": torch.randn(4, 32),
        "feat_bn": torch.randn(4, 32),
        "emb": torch.randn(4, 32),
        "logits": torch.randn(4, 3),
    }
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    total_loss, logs = criterion(outputs, labels)

    assert torch.is_tensor(total_loss)
    assert "loss/id" in logs
    assert torch.allclose(id_loss.last_logits, outputs["logits"])
    assert not torch.allclose(id_loss.last_logits, outputs["feat_raw"][:, :3])
    assert torch.equal(id_loss.last_labels, labels)
