"""Evaluation ownership with conflicting presets and synthetic local datasets."""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image

from reid.data.build import build_test_loader
from reid.models.build import build_model
from reid.utils.checkpoint import save_checkpoint
from reid.utils.config import load_config, validate_reid_config
from reid.utils.evaluation_config import prepare_standalone_config
from scripts.evaluate_cross_domain import build_cross_domain_config, validate_output_identity


DATASETS = ("market1501", "duke", "cuhk03", "msmt17")


def source_config(architecture="pcb", dataset="market1501"):
    cfg = load_config(f"configs/baseline_{dataset}_resnet50_triplet.yaml")
    cfg["data"]["root"] = "absent-source-data"
    cfg["system"].update(device="cpu", gpu_id=0)
    cfg["data"].update(num_workers=0, pin_memory=False)
    if architecture == "pcb":
        cfg["model"] = {"name": "pcb", "pretrained": False}
        cfg["data"]["test"]["images"]["size"] = [384, 128]
        cfg["loss"]["triplet"]["enabled"] = False
    else:
        cfg["model"]["backbone"]["pretrained"] = False
        cfg["model"]["head"]["embedding_dim"] = 32
    cfg["eval"]["weight"] = "source-selected-best.pth"
    validate_reid_config(cfg)
    return cfg


def target_config(dataset, root="target-root"):
    cfg = load_config(f"configs/baseline_{dataset}_resnet50_triplet.yaml")
    cfg["data"].update(root=str(root), num_workers=0, pin_memory=False)
    cfg["system"].update(device="cpu", gpu_id=2)
    cfg["data"]["test"]["batch"]["size"] = 2
    cfg["data"]["test"]["aug"].update(mean=[.1, .2, .3], std=[.5, .6, .7])
    cfg["data"]["test"]["dataset"]["num_classes"] = 999
    cfg["model"]["head"].update(num_classes=999, embedding_dim=64)
    cfg["eval"].update(weight="wrong-target-checkpoint.pth", normalize_feat=False,
                          distance="cosine", topk=[1])
    cfg["eval"]["rerank"]["enabled"] = True
    if dataset == "cuhk03":
        cfg["data"]["test"]["dataset"]["image_type"] = "labeled"
    return cfg


@pytest.mark.parametrize("architecture", ["pcb", "reid_baseline"])
def test_source_row_ownership_conflicts_and_copy_safety(architecture):
    source = source_config(architecture)
    source["system"].update(device="cuda", gpu_id=0)
    source["data"].update(num_workers=3, pin_memory=True)
    original = copy.deepcopy(source)
    for name in DATASETS:
        target = target_config(name)
        target["data"]["test"]["loader"]["shuffle"] = True
        target["system"]["amp"] = True
        if name != "msmt17":
            target["data"]["test"]["dataset"]["split"] = "val"
        before = copy.deepcopy(target)
        cfg = build_cross_domain_config(source, target, name, f"results/{architecture}_market_to_{name}")
        assert cfg["model"] == source["model"]
        assert cfg["eval"] == source["eval"]
        for field in ("images", "aug", "loader"):
            assert cfg["data"]["test"][field] == source["data"]["test"][field]
        assert cfg["data"]["test"]["dataset"] == target["data"]["test"]["dataset"]
        assert cfg["data"]["root"] == "target-root"
        assert cfg["data"]["test"]["batch"]["size"] == 2
        assert cfg["system"]["gpu_id"] == 2 and cfg["system"]["device"] == "cpu"
        assert cfg["system"]["amp"] is False
        assert cfg["data"]["num_workers"] == 0 and cfg["data"]["pin_memory"] is False
        for section in ("loss", "optim", "sched", "train"):
            assert cfg[section] == source[section]
        assert cfg["data"]["train"] == source["data"]["train"]
        cfg["model"]["name"] = "mutated"
        cfg["data"]["test"]["images"]["size"][0] = 99
        cfg["data"]["test"]["dataset"]["name"] = "mutated"
        cfg["eval"]["topk"].append(99)
        assert source == original and target == before


def test_reversal_and_diagonal_consistency():
    market, duke = source_config(), source_config("reid_baseline", "duke")
    duke["data"]["test"]["aug"]["mean"] = [.1, .2, .3]
    forward = build_cross_domain_config(market, duke, "duke", "market_to_duke")
    reverse = build_cross_domain_config(duke, market, "market1501", "duke_to_market")
    assert forward["model"] == market["model"]
    assert reverse["model"] == duke["model"]
    assert forward["data"]["test"]["images"]["size"] == [384, 128]
    assert reverse["data"]["test"]["images"]["size"] == [256, 128]
    assert forward["data"]["test"]["dataset"] == duke["data"]["test"]["dataset"]
    assert reverse["data"]["test"]["dataset"] == market["data"]["test"]["dataset"]
    assert forward["experiment"]["output_dir"] != reverse["experiment"]["output_dir"]
    diagonal = build_cross_domain_config(market, market, "market1501", market["experiment"]["output_dir"])
    assert diagonal == prepare_standalone_config({"cfg": market}, market) == market


@pytest.mark.parametrize("field,value", [
    ("model.name", "different"), ("model.variant", "different"),
    ("data.test.images.size", [256, 128]),
    ("data.test.aug.mean", [.1, .2, .3]), ("data.test.aug.std", [1., 1., 1.]),
    ("data.test.loader.shuffle", True), ("eval.normalize_feat", False),
    ("eval.distance", "cosine"), ("eval.rerank.enabled", True), ("eval.topk", [1]),
])
def test_standalone_rejects_source_contract_conflicts(field, value):
    source = source_config()
    request = copy.deepcopy(source)
    destination = request
    keys = field.split(".")
    for key in keys[:-1]:
        destination = destination[key]
    destination[keys[-1]] = value
    original = copy.deepcopy(request)
    with pytest.raises(ValueError, match="conflicts with checkpoint source"):
        prepare_standalone_config({"cfg": source}, request)
    assert request == original


def test_standalone_runtime_overrides_and_legacy_fallback():
    source = source_config()
    request = copy.deepcopy(source)
    request["system"].update(device="cuda", gpu_id=3)
    request["data"].update(root="relocated", num_workers=7, pin_memory=True)
    request["data"]["test"]["batch"]["size"] = 16
    request["experiment"]["output_dir"] = "new-eval-output"
    request["eval"]["weight"] = "explicit-selected.pth"
    cfg = prepare_standalone_config({"cfg": source}, request)
    assert cfg["model"] == source["model"]
    assert cfg["data"]["test"]["images"] == source["data"]["test"]["images"]
    assert cfg["system"]["device"] == "cuda" and cfg["system"]["gpu_id"] == 3
    assert cfg["data"]["num_workers"] == 7 and cfg["data"]["pin_memory"] is True
    assert cfg["data"]["test"]["batch"]["size"] == 16
    assert cfg["data"]["root"] == "relocated"
    assert cfg["experiment"]["output_dir"] == "new-eval-output"
    for checkpoint in ({}, {"cfg": None}):
        fallback = prepare_standalone_config(checkpoint, request)
        assert fallback == request and fallback is not request
    with pytest.raises(ValueError, match="mapping"):
        prepare_standalone_config({"cfg": []}, request)


def test_target_name_validation():
    with pytest.raises(ValueError, match="expected 'msmt17'"):
        build_cross_domain_config(source_config(), target_config("duke"), "msmt17", "out")


def write_target(root, name):
    """Two/three evaluation images only; no training partition or train labels."""
    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 8), (0, 0, 0)).save(path)

    if name == "msmt17":
        directory = root / "msmt17" / "MSMT17_V2"
        for kind, cam in (("query", 0), ("gallery", 1)):
            relative = f"0007/0007_0000_{cam}_0001.jpg"
            image(directory / "mask_test_v2" / relative)
            # Include camera 0 in gallery too: parser normalizes each list separately.
            rows = [f"{relative} 7"]
            if kind == "gallery":
                extra = "0008/0008_0000_0_0001.jpg"
                image(directory / "mask_test_v2" / extra)
                rows.append(f"{extra} 8")
            (directory / f"list_{kind}.txt").write_text("\n".join(rows) + "\n")
        return [7, 7, 8], [0, 1, 0], [0, 1, 1]
    directory = root / name
    if name == "cuhk03":
        directory /= "labeled"
    names = ["00000007_0001_00000000.jpg", "00000007_0002_00000000.jpg"]
    for filename in names:
        image(directory / "images" / filename)
    with (directory / "partitions.pkl").open("wb") as stream:
        pickle.dump({"test_im_names": names, "test_marks": [0, 1]}, stream)
    return [7, 7], [1, 2], [0, 1]


@pytest.mark.parametrize("name", DATASETS)
def test_real_test_loader_uses_source_pixels_and_target_protocol(tmp_path, name):
    source = source_config()
    root = tmp_path / name
    pids, cams, marks = write_target(root, name)
    target = target_config(name, root)
    cfg = build_cross_domain_config(source, target, name, tmp_path / "result")
    loader = build_test_loader(cfg)
    assert loader.batch_size == 2 and loader.num_workers == 0
    assert loader.drop_last is False
    batches = list(loader)
    images = torch.cat([b[0] for b in batches])
    assert images.shape == (len(pids), 3, 384, 128)
    assert torch.cat([b[1] for b in batches]).tolist() == pids
    assert torch.cat([b[2] for b in batches]).tolist() == cams
    assert torch.cat([b[4] for b in batches]).tolist() == marks
    aug = source["data"]["test"]["aug"]
    expected = -torch.tensor(aug["mean"]) / torch.tensor(aug["std"])
    torch.testing.assert_close(images, expected[None, :, None, None].expand_as(images), rtol=0, atol=0)
    if name == "cuhk03":
        assert loader.dataset.image_type == "labeled"
        assert loader.dataset.protocol == "processed_partition"
        assert loader.dataset.split_id is None


def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation must not train, adapt, initialize pretrained weights or save checkpoints")
    for target in ("reid.data.build.build_train_loader", "reid.optim.build.build_optimizer",
                   "reid.losses.build.build_criterion", "reid.utils.checkpoint.save_checkpoint",
                   "reid.models.pcb._read_historical_weights", "torch.hub.load_state_dict_from_url"):
        monkeypatch.setattr(target, forbidden)


@pytest.mark.parametrize("architecture", ["pcb", "reid_baseline"])
def test_real_cross_entry_point_reconstruction_row(tmp_path, monkeypatch, architecture):
    import scripts.evaluate_cross_domain as cross
    source = source_config(architecture)
    torch.manual_seed(15)
    model = build_model(source, num_classes=3).eval()
    path = tmp_path / "ckpt_best.pth"
    save_checkpoint(path, model, cfg=source, epoch=10)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    forbid_training(monkeypatch)
    real_reconstruct, real_evaluate = cross.reconstruct_model, cross.evaluate_reid
    targets = DATASETS if architecture == "pcb" else ("market1501", "duke")
    seen = []

    def reconstruct(checkpoint):
        rebuilt = real_reconstruct(checkpoint)
        assert rebuilt.checkpoint_metadata["num_classes"] == 3
        assert rebuilt.embedding_dim == (1536 if architecture == "pcb" else 32)
        if architecture == "pcb":
            assert all(head.out_features == 3 for head in rebuilt.classifiers.fc_list)
        else:
            assert rebuilt.classifier.out_features == 3
        for key, value in model.state_dict().items():
            assert torch.equal(value, rebuilt.state_dict()[key])
        return rebuilt

    def evaluate(cfg, rebuilt, loader, device):
        assert cfg["model"] == source["model"]
        assert cfg["eval"]["normalize_feat"] is True
        assert cfg["eval"]["distance"] == "euclidean"
        assert cfg["eval"]["rerank"]["enabled"] is False
        assert cfg["eval"]["topk"] == [1, 5, 10]
        assert cfg["eval"]["weight"] == str(path)
        assert cfg["data"]["test"]["images"] == source["data"]["test"]["images"]
        seen.append(cfg["data"]["test"]["dataset"]["name"])
        scores = real_evaluate(cfg, rebuilt, loader, device)
        assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in rebuilt.parameters())
        return scores

    monkeypatch.setattr(cross, "reconstruct_model", reconstruct)
    monkeypatch.setattr(cross, "evaluate_reid", evaluate)
    for name in targets:
        root = tmp_path / f"data-{name}"
        write_target(root, name)
        target = target_config(name, root)
        target_yaml = tmp_path / f"{name}.yaml"
        cross.save_yaml(target, target_yaml)
        output = tmp_path / f"{architecture}_market1501_to_{name}"
        args = argparse.Namespace(config=str(target_yaml), opts=["data.test.batch.size=1"],
                                  checkpoint=str(path), source_dataset="market1501",
                                  target_dataset=name, output_dir=str(output))
        monkeypatch.setattr(cross, "parse_args", lambda: args)
        cross.main()
        record = json.loads((output / "cross_dataset.json").read_text())
        assert (record["source_dataset"], record["target_dataset"]) == ("market1501", name)
        assert record["checkpoint_path"] == str(path)
        assert record["architecture"] == architecture
        saved = load_config(output / "config.resolved.yaml")
        assert saved == record["resolved_config"]
        assert saved["data"]["test"]["batch"]["size"] == 1
        assert saved["model"] == source["model"]
    assert seen == list(targets)
    first_output = tmp_path / f"{architecture}_market1501_to_{targets[0]}"
    previous = (first_output / "cross_dataset.json").read_bytes()
    args.output_dir = str(first_output)  # Last target must not overwrite the diagonal.
    with pytest.raises(ValueError, match="different source/target/checkpoint"):
        cross.main()
    assert (first_output / "cross_dataset.json").read_bytes() == previous
    assert seen == list(targets)  # Collision fails before evaluating again.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("legacy", [False, True])
def test_real_standalone_entry_point_source_config(tmp_path, monkeypatch, legacy):
    import scripts.evaluate as standalone
    import reid.engine.evaluator as evaluator
    import reid.utils.artifacts as artifacts
    source = source_config("reid_baseline" if legacy else "pcb")
    root = tmp_path / "dataset"
    write_target(root, "market1501")
    source["data"]["root"] = str(root)
    model = build_model(source, num_classes=3).eval()
    path = tmp_path / "best.pth"
    if legacy:
        torch.save(model.state_dict(), path)
    else:
        save_checkpoint(path, model, cfg=source, epoch=10)
    request = copy.deepcopy(source)
    output = tmp_path / "standalone"
    request["experiment"]["output_dir"] = str(output)
    request["data"]["test"]["batch"]["size"] = 1
    config_path = tmp_path / "request.yaml"
    from reid.utils.config import save_yaml
    save_yaml(request, config_path)
    forbid_training(monkeypatch)
    args = argparse.Namespace(config=str(config_path), weight=str(path), opts=[])
    monkeypatch.setattr(standalone, "parse_args", lambda: args)
    monkeypatch.setattr(artifacts, "save_run_artifacts", lambda *a, **k: {"command": "synthetic", "environment": "CPU"})
    real_evaluate = evaluator.evaluate_reid
    def evaluate(cfg, rebuilt, loader, device, logger=None):
        assert cfg["model"] == source["model"]
        assert cfg["eval"]["weight"] == str(path)
        assert rebuilt.checkpoint_metadata["num_classes"] == 3
        assert loader.batch_size == 1
        return real_evaluate(cfg, rebuilt, loader, device, logger=logger)
    monkeypatch.setattr(evaluator, "evaluate_reid", evaluate)
    standalone.main()
    saved = load_config(output / "config.resolved.yaml")
    assert saved["data"]["test"]["images"] == source["data"]["test"]["images"]
    assert saved["eval"]["weight"] == str(path)
    assert (output / "metrics" / "eval_best.json").is_file()


def test_output_identity_rejects_colliding_direction_or_checkpoint(tmp_path):
    record = {"source_dataset": "market1501", "target_dataset": "duke", "checkpoint_path": "source.pth"}
    (tmp_path / "cross_dataset.json").write_text(json.dumps(record))
    validate_output_identity(tmp_path, "market1501", "duke", "source.pth")
    for source, target, path in (("duke", "market1501", "source.pth"),
                                 ("market1501", "duke", "other.pth")):
        with pytest.raises(ValueError, match="different source/target/checkpoint"):
            validate_output_identity(tmp_path, source, target, path)
    assert json.loads((tmp_path / "cross_dataset.json").read_text()) == record


def test_standalone_conflict_fails_before_artifacts_or_loader(tmp_path, monkeypatch):
    import scripts.evaluate as standalone
    from reid.utils.config import save_yaml
    source = source_config()
    checkpoint = tmp_path / "source.pth"
    torch.save({"cfg": source}, checkpoint)
    request = copy.deepcopy(source)
    request["data"]["test"]["images"]["size"] = [256, 128]
    output = tmp_path / "must-not-be-created"
    request["experiment"]["output_dir"] = str(output)
    config_path = tmp_path / "conflict.yaml"
    save_yaml(request, config_path)
    args = argparse.Namespace(config=str(config_path), weight=str(checkpoint), opts=[])
    monkeypatch.setattr(standalone, "parse_args", lambda: args)
    def forbidden(*args, **kwargs):
        pytest.fail("Conflicting source input must fail before construction")
    monkeypatch.setattr("reid.data.build.build_test_loader", forbidden)
    monkeypatch.setattr("reid.utils.checkpoint.reconstruct_model", forbidden)
    with pytest.raises(ValueError, match="source data.test.images"):
        standalone.main()
    assert not output.exists()
