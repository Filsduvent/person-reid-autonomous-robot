"""Diagnostics semantics and bounded synthetic optimization-invariance checks."""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.engine.train_loop import classification_accuracy, train_one_epoch
from reid.losses.build import build_criterion
from reid.models.build import build_model


def criterion(smoothing=0.0):
    return build_criterion({"model": {}, "loss": {"id": {
        "enabled": True, "weight": 1.0, "label_smoothing": smoothing,
        "head_aggregation": "sum"}}}, 3, None)


def controlled_heads():
    # Labels [0,1]: independent accuracies 1, 1/2, 0; the last head
    # dominates averaged logits, whose predictions are BOTH wrong.
    return [torch.tensor([[2., 0., 0.], [0., 2., 0.]], requires_grad=True),
            torch.tensor([[2., 0., 0.], [2., 0., 0.]], requires_grad=True),
            torch.tensor([[0., 20., 0.], [20., 0., 0.]], requires_grad=True)]


def test_single_tensor_exact_accuracy_and_no_grad(monkeypatch):
    z = controlled_heads()[1]
    labels = torch.tensor([0, 1])
    original = torch.Tensor.argmax
    def checked_argmax(self, *a, **kw):
        assert not torch.is_grad_enabled()
        return original(self, *a, **kw)
    monkeypatch.setattr(torch.Tensor, "argmax", checked_argmax)
    assert classification_accuracy(z, labels) == {"acc/id": 0.5}
    assert z.grad is None
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("heads", [1, 2, 3, 6])
def test_generic_head_counts_and_disagreement(heads, container, monkeypatch):
    z = container((controlled_heads() * 2)[:heads])
    labels = torch.tensor([0, 1])
    expected = sum(([1., .5, 0.] * 2)[:heads]) / heads
    original = torch.Tensor.argmax
    def checked_argmax(self, *a, **kw):
        assert not torch.is_grad_enabled()
        return original(self, *a, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "argmax", checked_argmax)
        result = classification_accuracy(z, labels)
    assert result == {"acc/id_mean_heads": expected}
    assert all(head.grad is None for head in z)
    if heads in (3, 6):
        ensemble = (torch.stack(z).mean(0).argmax(1) == labels).float().mean().item()
        assert ensemble == 0.0
        assert result["acc/id_mean_heads"] == 0.5


@pytest.mark.parametrize("bad", [[], (), ((torch.zeros(2, 3),),),
    (torch.zeros(2, 3), "bad"), (torch.zeros(2, 3), torch.zeros(2, 4)),
    (torch.zeros(2, 3), torch.zeros(1, 3)), (torch.zeros(2, 3, 1),),
    (torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.float64)),
    (torch.zeros(2, 3), torch.zeros(2, 3, device="meta"))])
def test_malformed_logits_use_common_contract(bad):
    with pytest.raises(ValueError, match="logits|Logits"):
        classification_accuracy(bad, torch.tensor([0, 1]))


def test_absent_logits_and_label_contract():
    assert classification_accuracy(None, torch.tensor([0, 1])) == {}
    with pytest.raises(ValueError, match="logits"):
        criterion()({"logits": None}, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="batch"):
        classification_accuracy(controlled_heads(), torch.tensor([0]))
    with pytest.raises(ValueError, match="labels"):
        classification_accuracy(controlled_heads(), torch.tensor([[0], [1]]))


class TinyPlugin(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(4, 3) for _ in range(heads or 1))
        self.single = heads == 0

    def forward(self, x):
        z = tuple(head(x) for head in self.heads)
        return {"emb": x, "feat_raw": None, "feat_bn": None,
                "logits": z[0] if self.single else z}


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, key, value, global_step):
        self.values[key] = value


def assert_same_update(a, b, opt_a, opt_b):
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(opt_a.state[x]["momentum_buffer"],
                                   opt_b.state[y]["momentum_buffer"], rtol=0, atol=0)
    assert opt_a.state_dict()["param_groups"] == opt_b.state_dict()["param_groups"]


@pytest.mark.parametrize("heads", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_diagnostics_before_backward_preserve_exact_update(heads, smoothing):
    torch.manual_seed(42)
    a = TinyPlugin(heads)
    b = copy.deepcopy(a)
    x, labels = torch.randn(4, 4), torch.tensor([0, 1, 2, 0])
    opt_a = torch.optim.SGD(a.parameters(), lr=.01, momentum=.9)
    opt_b = torch.optim.SGD(b.parameters(), lr=.01, momentum=.9)
    out_a, out_b = a(x), b(x)
    loss_a, logs_a = criterion(smoothing)(out_a, labels)
    loss_b, logs_b = criterion(smoothing)(out_b, labels)
    state = {key: value.clone() for key, value in b.state_dict().items()}
    diagnostic = classification_accuracy(out_b["logits"], labels)
    assert all(isinstance(v, float) for v in diagnostic.values())
    assert all(torch.equal(state[k], v) for k, v in b.state_dict().items())
    assert all(p.grad is None for p in b.parameters())
    assert logs_a == logs_b
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    loss_a.backward()
    loss_b.backward()
    opt_a.step()
    opt_b.step()
    assert_same_update(a, b, opt_a, opt_b)


@pytest.mark.parametrize("heads", [0, 2, 3, 6])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_common_loop_logging_and_update_equal_manual_step(heads, smoothing, capsys):
    torch.manual_seed(7)
    manual = TinyPlugin(heads)
    actual = copy.deepcopy(manual)
    x, labels = torch.randn(4, 4), torch.tensor([0, 1, 2, 0])
    opt_a = torch.optim.SGD(manual.parameters(), lr=.01, momentum=.9)
    opt_b = torch.optim.SGD(actual.parameters(), lr=.01, momentum=.9)
    out = manual(x)
    loss, expected_logs = criterion(smoothing)(out, labels)
    zs = [out["logits"]] if heads == 0 else out["logits"]
    expected_accuracy = sum((z.argmax(1) == labels).float().mean().item() for z in zs) / len(zs)
    loss.backward()
    opt_a.step()
    writer = Writer()
    average = train_one_epoch(actual, [(x, labels)], criterion(smoothing), opt_b,
                              None, torch.device("cpu"), False, 1, tb_writer=writer)
    assert average == loss.item()
    assert_same_update(manual, actual, opt_a, opt_b)
    tag = "acc/id" if heads == 0 else "acc/id_mean_heads"
    assert {k for k in writer.values if k.startswith("acc/")} == {tag}
    assert writer.values[tag] == pytest.approx(expected_accuracy, abs=1e-7)
    for key, value in expected_logs.items():
        assert writer.values[key] == value
    assert writer.values["lr"] == writer.values["lr/base"] == .01
    assert f"{tag.replace('/', '_')}=" in capsys.readouterr().out


def test_real_pcb_diagnostic_loss_and_backward(monkeypatch):
    import reid.models.pcb as pcb
    def forbidden(*a, **kw):
        raise AssertionError("Unexpected initialization")
    monkeypatch.setattr(pcb, "_read_historical_weights", forbidden)
    torch.manual_seed(42)
    model = build_model({"model": {"name": "pcb", "pretrained": False}}, 3).train()
    out = model(torch.randn(2, 3, 384, 128))
    labels = torch.tensor([0, 1])
    loss, logs = criterion()(out, labels)
    expected = sum(F.cross_entropy(z, labels) for z in out["logits"])
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    diagnostic = classification_accuracy(out["logits"], labels)
    assert set(diagnostic) == {"acc/id_mean_heads"}
    assert 0 <= diagnostic["acc/id_mean_heads"] <= 1
    assert logs["loss/id"] == logs["loss/total"] == expected.item()
    assert out["feat_raw"] is out["feat_bn"] is model.feat_dim is None
    loss.backward()
    for group in [model.backbone, *model.reductions.local_conv_list, *model.classifiers.fc_list]:
        grads = [p.grad for p in group.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)
