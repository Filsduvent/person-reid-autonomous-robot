"""Generic CE arithmetic and real PCB backward; no optimizer/training-loop step."""
import pytest
import torch
import torch.nn.functional as F

from reid.losses.build import LossBundle, build_criterion
from reid.models.build import build_model
from reid.utils.config import validate_reid_config


def config(aggregation="sum", smoothing=0.0, weight=1.0):
    return {"model": {"name": "pcb", "pretrained": False}, "loss": {
        "id": {"enabled": True, "head_aggregation": aggregation,
               "label_smoothing": smoothing, "weight": weight},
        "triplet": {"enabled": False}, "center": {"enabled": False}}}


def outputs(logits):
    return {"logits": logits, "feat_raw": None, "feat_bn": None}


def reference_ce(z, labels, smoothing):
    if smoothing == 0:
        return F.cross_entropy(z, labels)
    # Historical smoothing formula, independently expressed as target probabilities.
    target = torch.full_like(z, smoothing / z.shape[1])
    target.scatter_(1, labels[:, None], 1 - smoothing + smoothing / z.shape[1])
    return (-target * F.log_softmax(z, dim=1)).sum(dim=1).mean()


@pytest.mark.parametrize("heads", [1, 2, 3, 6])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
@pytest.mark.parametrize("smoothing,weight", [(0.0, 1.0), (0.1, 2.3)])
@pytest.mark.parametrize("container", [tuple, list])
def test_multi_head_value_and_every_gradient(heads, aggregation, smoothing, weight, container):
    torch.manual_seed(42)
    actual_heads = container(torch.randn(4, 3, requires_grad=True) for _ in range(heads))
    reference_heads = [z.detach().clone().requires_grad_() for z in actual_heads]
    labels = torch.tensor([0, 1, 2, 0])
    criterion = build_criterion(config(aggregation, smoothing, weight), 3, None)
    actual, logs = criterion(outputs(actual_heads), labels)
    terms = [reference_ce(z, labels, smoothing) for z in reference_heads]
    expected_id = sum(terms)
    if aggregation == "mean":
        expected_id = expected_id / heads
    expected = weight * expected_id
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert logs["loss/id"] == pytest.approx(expected_id.item(), rel=1e-6)
    assert logs["loss/total"] == pytest.approx(expected.item(), rel=1e-6)
    assert set(logs) == {"loss/id", "loss/total", "loss/triplet", "loss/center"}
    actual.backward()
    expected.backward()
    for a, b in zip(actual_heads, reference_heads):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("weight", [1.0, 2.3])
@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_single_tensor_preserves_exact_value_and_gradient(smoothing, weight, aggregation):
    torch.manual_seed(7)
    z = torch.randn(4, 3, requires_grad=True)
    ref = z.detach().clone().requires_grad_()
    labels = torch.tensor([0, 1, 2, 0])
    actual, _ = build_criterion(config(aggregation, smoothing, weight), 3, None)(outputs(z), labels)
    expected = torch.zeros(()) + weight * reference_ce(ref, labels, smoothing)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(z.grad, ref.grad, rtol=0, atol=0)


def test_default_sum_preserves_six_ce_scale():
    cfg = config()
    del cfg["loss"]["id"]["head_aggregation"]
    validate_reid_config(cfg, num_classes=3)
    z = torch.zeros(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    loss, _ = build_criterion(cfg, 3, None)(outputs((z,) * 6), labels)
    torch.testing.assert_close(loss, 6 * F.cross_entropy(z, labels))


@pytest.mark.parametrize("bad", ["median", "SUM", "", None, [], 3])
def test_invalid_aggregation_rejected_by_config_and_constructor(bad):
    with pytest.raises(ValueError, match="head_aggregation"):
        validate_reid_config(config(bad), num_classes=3)
    with pytest.raises(ValueError, match="head_aggregation"):
        build_criterion(config(bad), 3, None)
    with pytest.raises(ValueError, match="head_aggregation"):
        LossBundle(head_aggregation=bad)


@pytest.mark.parametrize("bad", [
    (), [], ((torch.zeros(4, 3),),), (torch.zeros(4, 3), "bad"),
    (torch.zeros(4, 3), torch.zeros(4, 3, 1)),
    (torch.zeros(4, 3), torch.zeros(2, 3)),
    (torch.zeros(4, 3), torch.zeros(4, 2)),
    (torch.zeros(4, 3), torch.zeros(4, 3, dtype=torch.float64)),
    (torch.zeros(4, 3), torch.zeros(4, 3, device="meta")),
    (torch.zeros(4, 3, dtype=torch.long),),
    (torch.zeros(4, 0),), {"head": torch.zeros(4, 3)},
])
def test_malformed_heads_rejected(bad):
    with pytest.raises(ValueError, match="logits|Logits"):
        build_criterion(config(), 3, None)(outputs(bad), torch.tensor([0, 1, 2, 0]))


@pytest.mark.parametrize("labels", [torch.tensor([0, 1]), torch.zeros(4, 1, dtype=torch.long)])
def test_sequence_requires_compatible_labels(labels):
    with pytest.raises(ValueError, match="batch|targets"):
        build_criterion(config(), 3, None)(outputs((torch.zeros(4, 3),) * 2), labels)


def test_missing_logits_and_metric_requirements():
    labels = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="logits"):
        build_criterion(config(), 3, None)(outputs(None), labels)
    for metric in ("triplet", "center"):
        cfg = config()
        cfg["loss"][metric] = {"enabled": True, "margin": 0.3, "weight": 1.0}
        criterion = build_criterion(cfg, 3, 8)
        with pytest.raises(ValueError, match="feat_raw"):
            criterion(outputs((torch.zeros(4, 3),) * 2), labels)
        cfg["loss"]["id"]["enabled"] = False
        criterion = build_criterion(cfg, 3, 8)
        value, _ = criterion({"feat_raw": torch.randn(4, 8), "logits": None}, labels)
        assert torch.isfinite(value)
    with pytest.raises(ValueError, match="feat_dim"):
        build_criterion(cfg, 3, None)


def test_real_pcb_production_loss_backward(monkeypatch):
    import reid.models.pcb as pcb
    def forbidden(*a, **kw):
        raise AssertionError("Unexpected pretrained initialization")
    monkeypatch.setattr(pcb, "_read_historical_weights", forbidden)
    torch.manual_seed(42)
    cfg = config()
    validate_reid_config(cfg, num_classes=3)
    model = build_model(cfg, num_classes=3).train()
    criterion = build_criterion(cfg, 3, model.feat_dim)
    result = model(torch.randn(2, 3, 384, 128))
    assert result["feat_raw"] is result["feat_bn"] is model.feat_dim is None
    loss, logs = criterion(result, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    assert logs["loss/id"] == logs["loss/total"]
    loss.backward()
    groups = [model.backbone, *model.reductions.local_conv_list, *model.classifiers.fc_list]
    assert len(groups) == 13
    for group in groups:
        grads = [p.grad for p in group.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)
