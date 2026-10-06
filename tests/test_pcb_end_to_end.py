"""Synthetic lifecycle through scripts.train.main; never opens benchmark data.

Only data sources, scalar collection, and policy-test scores are injected.
Training, evaluation, persistence, reconstruction, and resume remain real.
The injected mAP sequence is orchestration evidence, not retrieval performance.
"""

import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from reid.data.collate import train_collate_fn, test_collate_fn
from reid.data.protocol import ReIDTrainSample, ReIDEvalSample
from reid.engine import evaluator
from reid.models import pcb
from reid.utils.checkpoint import reconstruct_model
from scripts import train


class SyntheticPeople(Dataset):
    def __init__(self, evaluation=False):
        self.evaluation = evaluation
        self.images = torch.randn(4, 3, 384, 128, generator=torch.Generator().manual_seed(18))
        self.labels, self.num_classes = [0, 1, 2, 0], 3
        self.pids, self.camids = [101, 102, 101, 102], [0, 0, 1, 1]
        self.marks = [0, 0, 1, 1]
        self.im_names = [f"synthetic_{i}.png" for i in range(4)]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        if self.evaluation:
            return ReIDEvalSample(self.images[index], self.pids[index], self.camids[index],
                                 self.im_names[index], self.marks[index])
        return ReIDTrainSample(self.images[index], self.labels[index])


def state_digest(value):
    """Compare complete nested model/optimizer/scheduler states without retaining copies."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


class ScalarRecorder:
    def __init__(self):
        self.values = []

    def add_scalar(self, tag, value, global_step):
        assert math.isfinite(float(value)), tag
        if tag == "acc/id_mean_heads":
            assert 0 <= float(value) <= 1
        self.values.append((tag, float(value), global_step))

    def close(self):
        pass


def test_pcb_shared_training_lifecycle(tmp_path, monkeypatch):
    root = tmp_path / "pcb_phase18_synthetic"
    root.mkdir()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    # Restore global settings changed by the actual command entry point.
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        _exercise_lifecycle(root, monkeypatch)
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark


def _exercise_lifecycle(root, monkeypatch):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/pcb/market1501.yaml").read_text())
    cfg["experiment"].update(name="pcb_phase18_synthetic", notes="Synthetic only; mAP injected for policy checks.")
    cfg["system"].update(device="cpu", amp=False, log_interval=1)
    cfg["repro"].update(deterministic=True, benchmark=False)
    cfg["model"]["pretrained"] = False
    cfg["data"].update(root=str(root / "NO_REAL_DATA"), num_workers=0, pin_memory=False)
    for split in ("train", "test"):
        cfg["data"][split]["dataset"]["name"] = "phase18_synthetic"
    cfg["data"]["train"]["batch"]["batch_size"] = 2
    cfg["data"]["test"]["batch"]["size"] = 2
    cfg["train"].update(epochs=6, eval_interval=1)
    cfg["sched"]["milestones"] = [2]

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic lifecycle attempted pretrained initialization/download")

    monkeypatch.setattr(pcb, "_read_historical_weights", forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    train_loader = DataLoader(SyntheticPeople(), batch_size=2, shuffle=True, drop_last=True,
                              generator=torch.Generator(), collate_fn=train_collate_fn)
    eval_loader = DataLoader(SyntheticPeople(True), batch_size=2, collate_fn=test_collate_fn)
    monkeypatch.setattr(train, "build_train_loader", lambda cfg: (train_loader, 3))
    monkeypatch.setattr(train, "build_test_loader", lambda cfg: eval_loader)
    recorder = ScalarRecorder()
    monkeypatch.setattr(train, "setup_tensorboard", lambda *args: recorder)
    real_epoch, real_save, real_eval = train.train_one_epoch, train.save_checkpoint, evaluator.evaluate_reid
    real_normalize, real_distance = evaluator.normalize, evaluator.compute_dist
    scores = [0.2, 0.8, 0.8, 0.3, 0.9, 0.4]
    evidence = {"updates": 0, "losses": [], "initial_lrs": [], "step_lrs": [],
                "best_writes": [], "raw_synthetic_metrics": [], "normalizations": 0}
    context = {"run": "continuous", "epoch": 0, "eval_calls": 0}
    states = {}
    saved_outputs = {}
    regions = ["backbone."] + [f"reductions.local_conv_list.{i}." for i in range(6)] + [
        f"classifiers.fc_list.{i}." for i in range(6)]

    def normalize(features, axis):
        assert features.shape == (4, 1536) and axis == 1
        evidence["normalizations"] += 1
        return real_normalize(features, axis=axis)

    def distance(query, gallery, metric):
        assert metric == "euclidean"
        np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(gallery, axis=1), 1, atol=1e-6)
        return real_distance(query, gallery, metric=metric)

    monkeypatch.setattr(evaluator, "normalize", normalize)
    monkeypatch.setattr(evaluator, "compute_dist", distance)

    def evaluate(config, model, loader, device, logger=None):
        before = evidence["normalizations"]
        result = real_eval(config, model, loader, device, logger=logger)
        assert evidence["normalizations"] == before + 1
        raw = {key: result[key] for key in ("mAP", "mINP", "Rank1", "Rank5", "Rank10")}
        assert all(value is not None and math.isfinite(value) and 0 <= value <= 1 for value in raw.values())
        evidence["raw_synthetic_metrics"].append(raw)
        context["eval_calls"] += 1
        final_call = context["eval_calls"] == (7 if context["run"] == "continuous" else 2)
        selected_epoch = (5 if context["run"] == "continuous" else 2) if final_call else context["epoch"]
        if final_call:
            assert state_digest(model.state_dict()) == states[(context["run"], selected_epoch)]["model"]
        # Real extraction/ranking above; only mAP is controlled for selection policy.
        return dict(result, mAP=scores[selected_epoch - 1])

    monkeypatch.setattr(evaluator, "evaluate_reid", evaluate)

    def epoch(**kwargs):
        model, optimizer, criterion = (kwargs[key] for key in ("model", "optimizer", "criterion"))
        ep = kwargs["epoch"]
        context["epoch"] = ep
        # Explicit epoch-local sampler/RNG control, not checkpoint RNG restoration.
        train_loader.generator.manual_seed(18000 + ep)
        torch.manual_seed(19000 + ep)
        assert kwargs["aux_optimizer"] is None and not kwargs["amp"]
        named = dict(model.named_parameters())
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(parameters) == len({id(p) for p in parameters}) == len(named)
        assert {id(p) for p in parameters} == {id(p) for p in named.values()}
        assert isinstance(optimizer, torch.optim.SGD)
        for group in optimizer.param_groups:
            assert group["momentum"] == 0.9 and not group["nesterov"]
            assert group["weight_decay"] == 0.0005
            for p in group["params"]:
                name = next(name for name, item in named.items() if item is p)
                initial = 0.01 if name.startswith("backbone.") else 0.1
                assert group["initial_lr"] == pytest.approx(initial)
        if context["run"] == "continuous" and ep == 1:
            evidence["initial_lrs"] = sorted({g["lr"] for g in optimizer.param_groups})
            evidence["parameter_tensors"] = len(parameters)
        if context["run"] == "resume":
            for key, obj in (("model", model), ("optimizer", optimizer), ("scheduler", kwargs["scheduler"])):
                assert state_digest(obj.state_dict()) == states[("continuous", 2)][key]
        representatives = {prefix: next(p for name, p in named.items() if name.startswith(prefix)) for prefix in regions}
        saved = {}

        def before_step(opt, args, kw):
            for name, p in named.items():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            for prefix in regions:
                assert any(p.grad.abs().sum() > 0 for name, p in named.items() if name.startswith(prefix)), prefix
            saved.update({prefix: p.detach().clone() for prefix, p in representatives.items()})
            evidence["step_lrs"].append(sorted({g["lr"] for g in opt.param_groups}))

        def after_step(opt, args, kw):
            for prefix, p in representatives.items():
                assert torch.isfinite(p).all() and not torch.equal(saved[prefix], p), prefix
            evidence["updates"] += 1

        def check_loss(module, inputs, result):
            output, labels = inputs
            assert output["emb"].shape == (2, 1536)
            assert len(output["logits"]) == 6
            assert all(head.shape == (2, 3) for head in output["logits"])
            expected = sum(F.cross_entropy(head, labels) for head in output["logits"])
            loss, logs = result
            torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
            assert set(logs) == {"loss/id", "loss/total", "loss/triplet", "loss/center"}
            assert logs["loss/triplet"] == logs["loss/center"] == 0
            assert all(math.isfinite(float(value)) for value in logs.values())
            assert logs["loss/id"] == pytest.approx(float(loss.detach()))
            assert logs["loss/total"] == pytest.approx(float(loss.detach()))
            evidence["losses"].append(float(loss.detach()))

        handles = [optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step),
                   criterion.register_forward_hook(check_loss)]
        try:
            result = real_epoch(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
        states[(context["run"], ep)] = {key: state_digest(obj.state_dict()) for key, obj in (
            ("model", model), ("optimizer", optimizer), ("scheduler", kwargs["scheduler"]))}
        if context["run"] == "resume":
            assert states[("resume", ep)] == states[("continuous", ep)]
        return result

    monkeypatch.setattr(train, "train_one_epoch", epoch)

    def save(**kwargs):
        payload = real_save(**kwargs)
        path, ep = Path(kwargs["path"]), kwargs["epoch"]
        if path.name == "ckpt_best.pth":
            evidence["best_writes"].append([context["run"], ep])
        elif kwargs["scores"] is not None:
            best = torch.load(path.with_name("ckpt_best.pth"), map_location="cpu")
            expected_best = [1, 2, 2, 2, 5, 5][ep - 1]
            assert best["epoch"] == expected_best
            assert state_digest(best["model"]) == states[(context["run"], expected_best)]["model"]
            assert state_digest(payload["model"]) == states[(context["run"], ep)]["model"]
            if context["run"] == "continuous" and ep == 2:
                shutil.copyfile(path, root / "resume_epoch2.pth")
            if context["run"] == "continuous" and ep == 6:
                model = kwargs["model"]
                assert not model.training  # The preceding real evaluation sets eval mode.
                with torch.no_grad():
                    saved_outputs.update(model(eval_loader.dataset.images[:2]))
        return payload

    monkeypatch.setattr(train, "save_checkpoint", save)

    def run(config, output):
        config["experiment"]["output_dir"] = str(output)
        path = root / f"{output.name}.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(sys, "argv", ["scripts/train.py", "--config", str(path)])
        # Each CLI invocation gets its own logger handlers in this single process.
        with monkeypatch.context() as local:
            local.setattr(logging.getLogger("reid.train"), "handlers", [])
            train.main()

    continuous = root / "continuous"
    run(cfg, continuous)
    last = torch.load(continuous / "checkpoints/ckpt_last.pth", map_location="cpu")
    assert last["epoch"] == 6
    assert last["reconstruction"] == {"schema_version": 1, "output_contract_version": 1,
        "model_name": "pcb", "variant": "independent_part_reduction", "num_classes": 3, "embedding_dim": 1536}
    assert last["center_optimizer"] is None
    fallback = copy.deepcopy(cfg)
    fallback["model"]["num_classes"] = 999
    restored = reconstruct_model(last, cfg=fallback).eval()
    with torch.no_grad():
        reloaded = restored(eval_loader.dataset.images[:2])
    assert len(reloaded["logits"]) == 6
    assert all(head.shape == (2, 3) for head in reloaded["logits"])
    for key in ("emb", "logits"):
        left = saved_outputs[key] if isinstance(saved_outputs[key], (list, tuple)) else [saved_outputs[key]]
        right = reloaded[key] if isinstance(reloaded[key], (list, tuple)) else [reloaded[key]]
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    features = evaluator.extract_features(restored, eval_loader, torch.device("cpu"))[0]
    assert features.shape == (4, 1536) and np.isfinite(features).all()
    assert not np.allclose(np.linalg.norm(features, axis=1), 1)
    del last, restored

    resume_cfg = copy.deepcopy(cfg)
    resume_cfg["train"]["epochs"] = 3
    resume_cfg["train"]["save"]["resume"] = str(root / "resume_epoch2.pth")
    resumed = root / "resume"
    (resumed / "checkpoints").mkdir(parents=True)
    shutil.copyfile(root / "resume_epoch2.pth", resumed / "checkpoints/ckpt_best.pth")
    context.update(run="resume", epoch=0, eval_calls=0)
    states[("resume", 2)] = states[("continuous", 2)]
    run(resume_cfg, resumed)
    assert evidence["updates"] == 14
    assert evidence["best_writes"] == [["continuous", 1], ["continuous", 2], ["continuous", 5]]
    assert evidence["step_lrs"][0] == pytest.approx([0.01, 0.1])
    assert evidence["step_lrs"][4] == pytest.approx([0.001, 0.01])
    tags = {tag for tag, _, _ in recorder.values}
    assert {"loss/id", "loss/total", "acc/id_mean_heads"} <= tags
    assert "acc/id" not in tags and not any("head_" in tag or "head/" in tag for tag in tags)
    for output, best_ep, last_ep in ((continuous, 5, 6), (resumed, 2, 3)):
        for filename, ep, checkpoint in (("final_test.json", best_ep, "ckpt_best.pth"),
                                         ("final_epoch_test.json", last_ep, "ckpt_last.pth"),
                                         ("latest_val.json", last_ep, "ckpt_last.pth")):
            payload = json.loads((output / "metrics" / filename).read_text())
            assert (payload["epoch"], payload["checkpoint"], payload["mAP"]) == (ep, checkpoint, scores[ep - 1])
            assert payload["dataset"] == "phase18_synthetic"
    evidence.update(embedding_shape=list(features.shape), regions_updated=regions,
                    controlled_resume="exact model/optimizer/scheduler state equality after epoch 3",
                    injected_selection_scores=scores, scalar_tags=sorted(tags),
                    mean_head_accuracy=[value for tag, value, _ in recorder.values if tag == "acc/id_mean_heads"],
                    reconstruction_tolerance={"rtol": 0, "atol": 0})
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2))
    # Keep small diagnostics, configs, plots and metric identity evidence only.
    for path in root.rglob("*.pth"):
        path.unlink()
    print("PHASE18_EVIDENCE=" + str(root / "evidence.json"))
