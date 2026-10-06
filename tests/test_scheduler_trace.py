"""Frozen PCB LR contract: synthetic update traces, never dataset training."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reid.engine.train_loop import train_one_epoch
from reid.models.build import build_model
from reid.optim.build import build_optimizer, build_scheduler
from reid.utils.checkpoint import load_checkpoint, save_checkpoint


def config():
    return {
        "optim": {
            "name": "sgd", "lr": .1, "momentum": .9, "nesterov": False,
            "weight_decay": .0005, "weight_decay_bias": .0005,
            "bias_lr_factor": 1.,
            "param_groups": [{"prefix": "backbone.", "lr_mult": .1}],
        },
        "sched": {
            "name": "warmup_multistep", "milestones": [40], "gamma": .1,
            "warmup_iters": 0, "warmup_factor": 1., "warmup_method": "linear",
        },
    }


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)

    def forward(self, x):
        z = self.backbone(x)
        return {"emb": z, "logits": self.head(z), "feat_raw": None, "feat_bn": None}


def setup(steps, model=None, cfg=None):
    model = TinyModel() if model is None else model
    cfg = config() if cfg is None else cfg
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=steps)
    return model, optimizer, scheduler


def rates(optimizer):
    return tuple(group["lr"] for group in optimizer.param_groups)


def expected(optimizer, decayed):
    # Independent decimal contract, not scheduler internals or its base_lrs.
    return tuple(
        (.001 if decayed else .01) if group["param_name"].startswith("backbone.")
        else (.01 if decayed else .1)
        for group in optimizer.param_groups
    )


def assert_rates(actual, wanted):
    # Only allow binary floating-point product rounding, far below a 10x error.
    assert actual == pytest.approx(wanted, rel=1e-14, abs=0)


def advance(optimizer, scheduler, count, gradients=False):
    used, stored = [], []
    hook = optimizer.register_step_pre_hook(lambda opt, args, kwargs: used.append(rates(opt)))
    try:
        for _ in range(count):
            if gradients:
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        param.grad = torch.full_like(param, .01)
            optimizer.step()
            scheduler.step()
            stored.append(rates(optimizer))
    finally:
        hook.remove()
    return used, stored


def assert_full_trace(optimizer, used, stored, steps):
    boundary = 40 * steps
    assert len(used) == len(stored) == 120 * steps
    backbone = next(i for i, g in enumerate(optimizer.param_groups)
                    if g["param_name"].startswith("backbone."))
    new = [i for i, g in enumerate(optimizer.param_groups)
           if not g["param_name"].startswith("backbone.")]
    assert new
    for update, (actual, following) in enumerate(zip(used, stored), start=1):
        assert_rates(actual, expected(optimizer, update > boundary))
        assert_rates(following, expected(optimizer, update >= boundary))
        for index in new:
            assert actual[index] / actual[backbone] == pytest.approx(10., rel=1e-14, abs=0)
    for index in range(len(optimizer.param_groups)):
        changes = [u for u in range(2, len(used) + 1)
                   if used[u - 1][index] != used[u - 2][index]]
        assert changes == [boundary + 1]
        assert used[boundary][index] / used[boundary - 1][index] == pytest.approx(.1, rel=1e-14)
    # Explicit acceptance table plus early/no-warmup and late/no-extra-decay probes.
    for update in (1, 2, 3, steps, steps + 1, boundary - 1, boundary,
                   boundary + 1, 60 * steps, 80 * steps, 100 * steps, 120 * steps):
        assert_rates(used[update - 1], expected(optimizer, update > boundary))


@pytest.mark.parametrize("steps", [1, 3, 7])
def test_constructor_and_complete_120_epoch_trace(steps):
    model = TinyModel()
    cfg = config()
    optimizer = build_optimizer(cfg, model)
    assert_rates(rates(optimizer), expected(optimizer, False))
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=steps)
    assert scheduler.last_epoch == 0
    assert scheduler.milestones == [40 * steps]
    assert scheduler.warmup_iters == 0
    assert_rates(rates(optimizer), expected(optimizer, False))
    used, stored = advance(optimizer, scheduler, 120 * steps)
    assert_full_trace(optimizer, used, stored, steps)
    assert scheduler.last_epoch == 120 * steps


@pytest.mark.parametrize("steps", [1, 3, 7])
@pytest.mark.parametrize("position", ["early", "before", "at", "next", "late"])
def test_checkpoint_resume_lr_trace(tmp_path, steps, position):
    completed = {"early": 10 * steps, "before": 40 * steps - 1,
                 "at": 40 * steps, "next": 40 * steps + 1, "late": 60 * steps}[position]
    model, optimizer, scheduler = setup(steps)
    prefix, _ = advance(optimizer, scheduler, completed, gradients=True)
    path = tmp_path / "state.pth"
    save_checkpoint(path, model, optimizer, scheduler, epoch=completed // steps, cfg=config())
    restored_model, restored_optimizer, restored_scheduler = setup(steps)
    load_checkpoint(path, restored_model, restored_optimizer, restored_scheduler)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    for key, state in optimizer.state_dict()["state"].items():
        assert torch.equal(state["momentum_buffer"],
                           restored_optimizer.state_dict()["state"][key]["momentum_buffer"])
    assert rates(restored_optimizer) == rates(optimizer)
    assert_rates(rates(restored_optimizer), expected(optimizer, completed >= 40 * steps))
    remaining = 120 * steps - completed
    continuous, continuous_stored = advance(optimizer, scheduler, remaining, gradients=True)
    resumed, resumed_stored = advance(restored_optimizer, restored_scheduler, remaining, gradients=True)
    assert resumed == continuous  # Exact serialized LR-state continuation.
    assert resumed_stored == continuous_stored
    for update, actual in enumerate(prefix + resumed, start=1):
        assert_rates(actual, expected(optimizer, update > 40 * steps))
    assert restored_scheduler.last_epoch == 120 * steps


@pytest.mark.parametrize("steps", [3, 7])
def test_legacy_step_uses_raw_iterations_and_41_is_one_epoch_late(steps):
    cfg = config()
    cfg["sched"]["name"] = "step"
    _, optimizer, scheduler = setup(steps, cfg=cfg)
    assert list(scheduler.milestones.elements()) == [40]
    used, _ = advance(optimizer, scheduler, 41)
    assert_rates(used[39], expected(optimizer, False))
    assert_rates(used[40], expected(optimizer, True))  # Earlier than epoch 41.
    cfg = config()
    cfg["sched"]["milestones"] = [41]  # Negative control, never a selected recipe.
    _, optimizer, scheduler = setup(steps, cfg=cfg)
    used, _ = advance(optimizer, scheduler, 41 * steps + 1)
    assert_rates(used[40 * steps], expected(optimizer, False))  # Violates frozen boundary.
    assert_rates(used[41 * steps], expected(optimizer, True))


def test_common_loop_logs_next_lr_at_milestone(capsys):
    torch.manual_seed(14)
    loader = [(torch.ones(2, 2), torch.tensor([0, 1])) for _ in range(3)]
    steps = len(loader)
    model, optimizer, scheduler = setup(steps)
    advance(optimizer, scheduler, 39 * steps)
    used, logged = [], {}

    class Writer:
        def add_scalar(self, tag, value, global_step):
            logged[global_step, tag] = value

    def criterion(outputs, labels):
        loss = F.cross_entropy(outputs["logits"], labels)
        return loss, {"loss/total": loss.item()}

    hook = optimizer.register_step_pre_hook(lambda opt, args, kwargs: used.append(rates(opt)))
    try:
        for epoch in (40, 41):
            train_one_epoch(model, loader, criterion, optimizer, None, torch.device("cpu"),
                            amp=False, log_interval=1, scheduler=scheduler,
                            tb_writer=Writer(), epoch=epoch)
    finally:
        hook.remove()
    for index, actual in enumerate(used):
        assert_rates(actual, expected(optimizer, index >= steps))
    for epoch in (40, 41):
        for step in range(1, steps + 1):
            decayed = epoch == 41 or step == steps
            for group, value in zip(optimizer.param_groups, expected(optimizer, decayed)):
                assert_rates([logged[epoch * 100000 + step, "lr/" + group["group_name"]]], [value])
    boundary_line = next(line for line in capsys.readouterr().out.splitlines()
                         if "Epoch [40] Iter [3/3]" in line)
    assert "lr/prefix/backbone./regular=0.001" in boundary_line
    assert "lr/default/regular=0.01" in boundary_line


@pytest.mark.parametrize("steps", [1, 3, 7])
def test_real_pcb_groups_complete_trace_without_forward(monkeypatch, steps):
    def forbidden(*args, **kwargs):
        raise AssertionError("No pretrained initialization or PCB forward in scheduler verification")

    monkeypatch.setattr("reid.models.pcb._read_historical_weights", forbidden)
    model = build_model({"model": {"name": "pcb", "pretrained": False}}, num_classes=3)
    monkeypatch.setattr(model, "forward", forbidden)
    _, optimizer, scheduler = setup(steps, model=model)
    assert len(optimizer.param_groups) == 195
    assert sum(g["param_name"].startswith("backbone.") for g in optimizer.param_groups) == 159
    assert_rates(rates(optimizer), expected(optimizer, False))
    used, stored = advance(optimizer, scheduler, 120 * steps)
    assert_full_trace(optimizer, used, stored, steps)
    assert all(param.grad is None for param in model.parameters())
    assert not optimizer.state  # No momentum buffers or weight updates without gradients.
