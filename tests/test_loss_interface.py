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


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('aggregation', ['sum', 'mean'])
def test_model_metric_loss_matches_independent_scalar_and_gradient_oracle(count, aggregation):
    import torch.nn.functional as F
    torch.manual_seed(14)
    cfg = _cfg(center=False)
    cfg['loss']['triplet'].update(margin=1.2, weight=2.3, feature_source='model', feature_aggregation=aggregation)
    cfg['loss']['id'].update(weight=0.7, label_smoothing=0.0, head_aggregation='mean')
    labels = torch.tensor([0, 0, 1, 1])
    features = [torch.randn(4, d, requires_grad=True) for d in (3, 5, 7)[:count]]
    heads = [torch.randn(4, 3, requires_grad=True) for _ in range(8)]
    criterion = build_criterion(cfg, num_classes=3, feat_dim=None)
    loss, logs = criterion({'metric_features': features, 'logits': heads}, labels)
    copies = [f.detach().clone().requires_grad_() for f in features]
    head_copies = [h.detach().clone().requires_grad_() for h in heads]
    triplets = []
    for feature in copies:
        rows = []
        for i in range(4):
            distances = [(feature[i]-feature[j]).square().sum().clamp_min(1e-12).sqrt() for j in range(4)]
            positive = torch.stack([d for j, d in enumerate(distances) if labels[i] == labels[j]]).max()
            negative = torch.stack([d for j, d in enumerate(distances) if labels[i] != labels[j]]).min()
            rows.append((positive-negative+1.2).clamp_min(0))
        triplets.append(torch.stack(rows).mean())
    metric = sum(triplets) / (count if aggregation == 'mean' else 1)
    ce = sum(F.cross_entropy(h, labels) for h in head_copies) / 8
    expected = 2.3*metric + 0.7*ce
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-6)
    actual_grads = torch.autograd.grad(loss, features+heads)
    expected_grads = torch.autograd.grad(expected, copies+head_copies)
    for actual, wanted in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-6)
    assert logs['loss/triplet'] == pytest.approx(metric.item(), abs=2e-6)
    assert logs['loss/id'] == pytest.approx(ce.item(), abs=2e-6)
    for i, triplet in enumerate(triplets):
        assert logs[f'loss/triplet/feature_{i}'] == pytest.approx(triplet.item(), abs=2e-6)


@pytest.mark.parametrize('value', [None, [], ()])
def test_model_metric_enabled_requires_features_without_legacy_fallback(value):
    cfg = _cfg(id_loss=False, center=False)
    cfg['loss']['triplet']['feature_source'] = 'model'
    criterion = build_criterion(cfg, num_classes=None, feat_dim=None)
    with pytest.raises(ValueError, match='metric_features'):
        criterion({'metric_features': value, 'feat_raw': torch.randn(4, 8)}, torch.tensor([0, 0, 1, 1]))


def test_model_metric_center_still_uses_selected_legacy_tensor():
    cfg = _cfg(id_loss=False)
    cfg['loss']['triplet']['feature_source'] = 'model'
    criterion = build_criterion(cfg, num_classes=3, feat_dim=8)
    recorder = _RecordLoss()
    criterion.center_loss = recorder
    output = _outputs()
    output['metric_features'] = [torch.randn(4, 3, requires_grad=True)]
    criterion(output, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(recorder.last_x, output['feat_raw'])
    output['feat_raw'] = None
    with pytest.raises(ValueError, match='feat_raw'):
        criterion(output, torch.tensor([0, 0, 1, 1]))


def test_disabled_model_metric_allows_zero_and_legacy_logs_remain_exact():
    cfg = _cfg(triplet=False, center=False)
    cfg['loss']['triplet']['feature_source'] = 'model'
    _, logs = build_criterion(cfg, 3, None)({'logits': torch.randn(4, 3), 'metric_features': []}, torch.tensor([0, 0, 1, 1]))
    assert set(logs) == {'loss/total', 'loss/id', 'loss/triplet', 'loss/center'}


@pytest.mark.parametrize('key,value', [('feature_source', 'bad'), ('feature_aggregation', 'bad')])
def test_triplet_contract_choices_rejected(key, value):
    cfg = _cfg(center=False)
    cfg['loss']['triplet'][key] = value
    with pytest.raises(ValueError, match=key):
        build_criterion(cfg, 3, 8)


def test_model_metric_labels_device_and_batch_validation():
    cfg = _cfg(id_loss=False, center=False)
    cfg['loss']['triplet']['feature_source'] = 'model'
    criterion = build_criterion(cfg, None, None)
    with pytest.raises(ValueError, match='device'):
        criterion({'metric_features': torch.empty(4, 3, device='meta')}, torch.tensor([0, 0, 1, 1]))
    with pytest.raises(ValueError, match='batch'):
        criterion({'metric_features': torch.randn(3, 3)}, torch.tensor([0, 0, 1, 1]))


@pytest.mark.parametrize('key', ['feature_source', 'feature_aggregation'])
def test_config_rejects_invalid_triplet_contract(key):
    from reid.utils.config import validate_reid_config
    cfg = _cfg(center=False)
    cfg['model']['head']['classifier'] = True
    cfg['loss']['triplet'][key] = 'invalid'
    with pytest.raises(ValueError, match=key):
        validate_reid_config(cfg)
