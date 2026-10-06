"""Optional prefix LR policy; bounded synthetic steps, no recipe training."""
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.engine.train_loop import classification_accuracy, learning_rate_metrics, train_one_epoch
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim.build import build_optimizer, build_scheduler
from reid.utils.config import validate_reid_config


def config(rules=True, bias=1.0, name="sgd"):
    cfg = {"model": {}, "loss": {}, "optim": {
        "name": name, "lr": .1, "momentum": .9, "nesterov": False,
        "weight_decay": .0005, "weight_decay_bias": .0005,
        "bias_lr_factor": bias}}
    if rules:
        cfg["optim"]["param_groups"] = [{"prefix": "encoder.", "lr_mult": .1}]
    return cfg


class Plugin(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.head = nn.Linear(3, 3)
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        z = self.encoder(x)
        return {"emb": z, "logits": self.head(z), "feat_raw": None, "feat_bn": None}


def assert_coverage(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(actual) == len(set(actual)) == len(expected)
    assert set(actual) == expected


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
def test_prefix_composes_bias_decay_and_exact_coverage(name):
    model = Plugin()
    cfg = config(bias=2, name=name)
    cfg["optim"]["weight_decay_bias"] = .007
    original = copy.deepcopy(cfg)
    validate_reid_config(cfg)
    optimizer = build_optimizer(cfg, model)
    assert cfg == original
    assert_coverage(model, optimizer)
    for group in optimizer.param_groups:
        name = group["param_name"]
        mult = .1 if name.startswith("encoder.") else 1.
        assert group["lr_mult"] == mult
        assert group["lr"] == pytest.approx(.1 * mult * (2 if "bias" in name else 1))
        assert group["weight_decay"] == (.007 if "bias" in name else .0005)
        owner = "prefix/encoder." if mult == .1 else "default"
        assert group["group_name"] == owner + ("/bias" if "bias" in name else "/regular")
        assert group["prefix"] == ("encoder." if mult == .1 else None)


@pytest.mark.parametrize("rules", [None, {}, (), "encoder.", [None], [{}],
    [{"prefix": "encoder."}], [{"lr_mult": .1}],
    [{"prefix": "", "lr_mult": .1}], [{"prefix": " encoder.", "lr_mult": .1}],
    [{"prefix": 2, "lr_mult": .1}],
    [{"prefix": "encoder.", "lr_mult": .1, "weight_decay": 0}],
    [{"prefix": "encoder.", "lr_mult": .1}, {"prefix": "encoder.", "lr_mult": 2}],
])
def test_invalid_rule_structure_fails_config_and_builder(rules):
    cfg = config()
    cfg["optim"]["param_groups"] = rules
    with pytest.raises(ValueError, match="param_groups"):
        validate_reid_config(cfg)
    with pytest.raises(ValueError, match="param_groups"):
        build_optimizer(cfg, Plugin())


@pytest.mark.parametrize("mult", [0, -1, True, "0.1", None, float("nan"), float("inf"), -float("inf")])
def test_invalid_multiplier_fails(mult):
    cfg = config()
    cfg["optim"]["param_groups"][0]["lr_mult"] = mult
    with pytest.raises(ValueError, match="lr_mult"):
        validate_reid_config(cfg)
    with pytest.raises(ValueError, match="lr_mult"):
        build_optimizer(cfg, Plugin())


@pytest.mark.parametrize("prefix", ["encoders.", "frozen"])
def test_zero_trainable_match_rejected(prefix):
    cfg = config()
    cfg["optim"]["param_groups"][0]["prefix"] = prefix
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(cfg, Plugin())


def test_overlapping_matches_rejected():
    cfg = config()
    cfg["optim"]["param_groups"].append({"prefix": "encoder.weight", "lr_mult": .2})
    with pytest.raises(ValueError, match="Overlapping.*encoder.weight"):
        build_optimizer(cfg, Plugin())


def historical_optimizer(cfg, model):
    # Frozen pre-Phase-13 per-parameter layout, arithmetic, metadata and ordering.
    c = cfg["optim"]
    groups = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.append({"params": [p], "param_name": name,
                           "lr": c["lr"] * c["bias_lr_factor"] if "bias" in name else c["lr"],
                           "weight_decay": c["weight_decay_bias"] if "bias" in name else c["weight_decay"]})
    if c["name"] == "sgd":
        return torch.optim.SGD(groups, lr=c["lr"], momentum=c["momentum"], nesterov=c["nesterov"])
    cls = torch.optim.Adam if c["name"] == "adam" else torch.optim.AdamW
    return cls(groups, lr=c["lr"])


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("name", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("empty", [False, True])
def test_no_rules_exact_historical_layout_update_and_resume(name, empty):
    torch.manual_seed(42)
    a, cfg = Plugin(), config(False, bias=2, name=name)
    b = copy.deepcopy(a)
    if empty:
        cfg["optim"]["param_groups"] = []
    old, new = historical_optimizer(cfg, a), build_optimizer(cfg, b)
    assert_state_equal(old.state_dict(), new.state_dict())
    for model in (a, b):
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
    old.step()
    new.step()
    assert_state_equal(a.state_dict(), b.state_dict())
    assert_state_equal(old.state_dict(), new.state_dict())
    restored = build_optimizer(cfg, b)
    restored.load_state_dict(copy.deepcopy(old.state_dict()))
    assert_state_equal(old.state_dict(), restored.state_dict())
    assert learning_rate_metrics(new) == {"lr": .1, "lr/base": .1, "lr/bias": .2}


def test_real_baseline_no_rule_group_order_and_metadata():
    cfg = config(False, bias=2, name="adam")
    cfg["model"] = {"name": "reid_baseline", "backbone": {"pretrained": False, "last_conv_stride": 1},
                    "head": {"embedding_dim": 32, "bnneck": True, "normalize": True, "classifier": True}}
    model = build_model(cfg, 3)
    old, new = historical_optimizer(cfg, model), build_optimizer(cfg, model)
    assert_state_equal(old.state_dict(), new.state_dict())
    assert [id(g["params"][0]) for g in old.param_groups] == [id(g["params"][0]) for g in new.param_groups]


def test_controlled_sgd_updates_momentum_and_group_state_roundtrip(tmp_path):
    model = Plugin().double()
    cfg = config(bias=2)
    cfg["optim"]["weight_decay_bias"] = .007
    optimizer = build_optimizer(cfg, model)
    for p in model.parameters():
        p.data.fill_(1.)
    expected = {g["param_name"]: torch.ones_like(g["params"][0]) for g in optimizer.param_groups}
    buffers = {}
    for _ in range(2):  # exactly two synthetic updates, no forward/data loop
        for group in optimizer.param_groups:
            p, name = group["params"][0], group["param_name"]
            p.grad = torch.full_like(p, 2.)
            direction = 2. + group["weight_decay"] * expected[name]
            buffers[name] = .9 * buffers[name] + direction if name in buffers else direction
            expected[name] = expected[name] - group["lr"] * buffers[name]
        optimizer.step()
        for group in optimizer.param_groups:
            p, name = group["params"][0], group["param_name"]
            torch.testing.assert_close(p, expected[name], rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], buffers[name], rtol=1e-12, atol=1e-12)
    assert model.frozen.item() == 1.
    path = tmp_path / "optimizer.pth"
    torch.save(optimizer.state_dict(), path)
    restored = build_optimizer(cfg, model)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert_state_equal(optimizer.state_dict(), restored.state_dict())
    assert learning_rate_metrics(restored) == learning_rate_metrics(optimizer)
    cfg["sched"] = {"name": "warmup_multistep", "milestones": [40], "warmup_iters": 0}
    before = learning_rate_metrics(restored)
    scheduler = build_scheduler(cfg, restored, steps_per_epoch=1)
    assert scheduler is not None
    assert learning_rate_metrics(restored) == before  # construction only; Phase 14 owns the trace


@pytest.mark.parametrize("rules", [False, True])
@pytest.mark.parametrize("bias", [1., 2.])
def test_lr_logging_uses_current_group_metadata(rules, bias, capsys):
    cfg, model = config(rules, bias), Plugin()
    cfg["loss"] = {"id": {"enabled": True, "weight": 1., "label_smoothing": .1}}
    optimizer = build_optimizer(cfg, model)
    expected = ({"lr/prefix/encoder./regular": .01, "lr/prefix/encoder./bias": .01 * bias,
                 "lr/default/regular": .1, "lr/default/bias": .1 * bias} if rules
                else {"lr": .1, "lr/base": .1, **({"lr/bias": .2} if bias == 2 else {})})
    class Writer:
        def __init__(self):
            self.values = {}
        def add_scalar(self, key, value, global_step):
            self.values[key] = value
    writer = Writer()
    train_one_epoch(model, [(torch.ones(2, 3), torch.tensor([0, 1]))],
                    build_criterion(cfg, 3, None), optimizer, None,
                    torch.device("cpu"), False, 1, tb_writer=writer)
    actual = {k: v for k, v in writer.values.items() if k == "lr" or k.startswith("lr/")}
    assert actual == pytest.approx(expected)
    console = capsys.readouterr().out
    for key in expected:
        if key != "lr/base":
            assert f"{key}=" in console
    if rules:
        assert "lr/bias=" not in console
        assert "lr/base" not in actual and "lr" not in actual
    for group in optimizer.param_groups:
        group["lr"] *= .5
    assert learning_rate_metrics(optimizer) == pytest.approx({k: v * .5 for k, v in expected.items()})


def test_real_pcb_group_ownership_loss_diagnostic_and_bounded_step(monkeypatch):
    import reid.models.pcb as pcb
    def forbidden(*a, **kw):
        raise AssertionError("Unexpected pretrained initialization")
    monkeypatch.setattr(pcb, "_read_historical_weights", forbidden)
    cfg = config()
    cfg["model"] = {"name": "pcb", "pretrained": False}
    cfg["optim"]["param_groups"][0]["prefix"] = "backbone."
    cfg["loss"] = {"id": {"enabled": True, "weight": 1., "label_smoothing": 0., "head_aggregation": "sum"},
                   "triplet": {"enabled": False}, "center": {"enabled": False}}
    validate_reid_config(cfg, 3)
    torch.manual_seed(42)
    model = build_model(cfg, 3).train()
    optimizer = build_optimizer(cfg, model)
    assert_coverage(model, optimizer)
    groups = {g["param_name"]: g for g in optimizer.param_groups}
    for name, param in model.named_parameters():
        group = groups[name]
        assert group["params"][0] is param
        assert group["lr"] == pytest.approx(.01 if name.startswith("backbone.") else .1)
        assert group["weight_decay"] == .0005
        assert group["momentum"] == .9 and group["nesterov"] is False
    assert len(groups) == 195
    assert sum(n.startswith("backbone.") for n in groups) == 159
    for i in range(6):
        for layer in (0, 1):  # actual reduction Conv and BN names, weight and bias
            for kind in ("weight", "bias"):
                assert groups[f"reductions.local_conv_list.{i}.{layer}.{kind}"]["lr"] == .1
        for kind in ("weight", "bias"):
            assert groups[f"classifiers.fc_list.{i}.{kind}"]["lr"] == .1
    assert groups["backbone.bn1.weight"]["lr"] == pytest.approx(.01)
    assert groups["backbone.bn1.bias"]["lr"] == pytest.approx(.01)
    out = model(torch.randn(2, 3, 384, 128))
    labels = torch.tensor([0, 1])
    loss, logs = build_criterion(cfg, 3, model.feat_dim)(out, labels)
    torch.testing.assert_close(loss, sum(F.cross_entropy(z, labels) for z in out["logits"]), rtol=0, atol=0)
    assert logs["loss/id"] == logs["loss/total"] == loss.item()
    acc = classification_accuracy(out["logits"], labels)
    assert set(acc) == {"acc/id_mean_heads"} and 0 <= acc["acc/id_mean_heads"] <= 1
    loss.backward()
    p = model.backbone.conv1.weight
    before = p.detach().clone()
    expected_direction = p.grad + .0005 * before
    optimizer.step()  # one bounded real-PCB synthetic update; no epochs
    torch.testing.assert_close(p, before - .01 * expected_direction, rtol=1e-6, atol=1e-7)
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert torch.isfinite(param).all()
        assert torch.isfinite(optimizer.state[param]["momentum_buffer"]).all()
