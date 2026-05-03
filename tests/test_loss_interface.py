import copy

import pytest
import torch
import torch.nn as nn

from reid.losses.build import build_criterion


BASE_CFG = {
    "model": {
        "head": {
            "metric_feat": "raw",
        },
    },
    "loss": {
        "triplet": {
            "enabled": True,
            "margin": 0.3,
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
}


def _cfg(*, metric_feat="raw", triplet=True, id_loss=True, center=True):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["model"]["head"]["metric_feat"] = metric_feat
    cfg["loss"]["triplet"]["enabled"] = triplet
    cfg["loss"]["id"]["enabled"] = id_loss
    cfg["loss"]["center"]["enabled"] = center
    return cfg


def _outputs():
    return {
        "feat_raw": torch.randn(4, 8, requires_grad=True),
        "feat_bn": torch.randn(4, 8, requires_grad=True),
        "emb": torch.randn(4, 8, requires_grad=True),
        "logits": torch.randn(4, 3, requires_grad=True),
    }


class _RecordLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_x = None
        self.last_labels = None

    def forward(self, x, labels):
        self.last_x = x.detach().clone()
        self.last_labels = labels.detach().clone()
        return x.sum() * 0.0


class _RecordIDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_logits = None
        self.last_labels = None

    def forward(self, logits, labels):
        self.last_logits = logits.detach().clone()
        self.last_labels = labels.detach().clone()
        return logits.sum() * 0.0


def test_loss_builder_is_model_agnostic_and_accepts_minimal_head_config():
    cfg = _cfg()

    criterion = build_criterion(cfg, num_classes=3, feat_dim=8)

    assert criterion.metric_feat_key == "feat_raw"
    assert criterion.triplet is not None
    assert criterion.id_loss is not None
    assert criterion.center_loss is not None


def test_criterion_returns_scalar_loss_and_standard_logs_from_output_dict():
    criterion = build_criterion(_cfg(), num_classes=3, feat_dim=8)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss, logs = criterion(_outputs(), labels)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert set(logs) >= {"loss/total", "loss/triplet", "loss/id", "loss/center"}
    assert all(isinstance(value, float) for value in logs.values())


@pytest.mark.parametrize(
    ("metric_feat", "expected_key"),
    [
        ("raw", "feat_raw"),
        ("bn", "feat_bn"),
    ],
)
def test_triplet_and_center_use_configured_metric_feature(metric_feat, expected_key):
    criterion = build_criterion(_cfg(metric_feat=metric_feat, triplet=True, id_loss=False, center=True), num_classes=3, feat_dim=8)
    triplet = _RecordLoss()
    center = _RecordLoss()
    criterion.triplet = triplet
    criterion.center_loss = center
    criterion.center = center
    outputs = _outputs()
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss, logs = criterion(outputs, labels)

    assert torch.is_tensor(loss)
    assert "loss/triplet" in logs
    assert "loss/center" in logs
    assert torch.allclose(triplet.last_x, outputs[expected_key].detach())
    assert torch.allclose(center.last_x, outputs[expected_key].detach())
    assert torch.equal(triplet.last_labels, labels)
    assert torch.equal(center.last_labels, labels)


def test_id_loss_uses_logits_and_not_metric_features():
    criterion = build_criterion(_cfg(triplet=False, id_loss=True, center=False), num_classes=3, feat_dim=8)
    id_loss = _RecordIDLoss()
    criterion.id_loss = id_loss
    outputs = _outputs()
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    loss, logs = criterion(outputs, labels)

    assert torch.is_tensor(loss)
    assert "loss/id" in logs
    assert torch.allclose(id_loss.last_logits, outputs["logits"].detach())
    assert not torch.allclose(id_loss.last_logits, outputs["feat_raw"][:, :3].detach())
    assert torch.equal(id_loss.last_labels, labels)


def test_metric_losses_require_selected_feature_key():
    criterion = build_criterion(_cfg(metric_feat="bn", triplet=True, id_loss=False, center=False), num_classes=3, feat_dim=8)
    outputs = _outputs()
    outputs["feat_bn"] = None

    with pytest.raises(ValueError, match="feat_bn"):
        criterion(outputs, torch.tensor([0, 0, 1, 1], dtype=torch.long))


def test_id_loss_requires_logits_key():
    criterion = build_criterion(_cfg(triplet=False, id_loss=True, center=False), num_classes=3, feat_dim=8)
    outputs = _outputs()
    outputs["logits"] = None

    with pytest.raises(ValueError, match="logits"):
        criterion(outputs, torch.tensor([0, 1, 2, 0], dtype=torch.long))


def test_loss_builder_rejects_model_specific_metric_feature_values():
    cfg = _cfg(metric_feat="resnet_layer4")

    with pytest.raises(ValueError, match="Unsupported metric feature"):
        build_criterion(cfg, num_classes=3, feat_dim=8)
