"""Generic reconstruction contracts, including surrogate and real PCB round trips."""
import copy

import pytest
import torch
from torch import nn

import reid.models.build as builders
from reid.utils.checkpoint import (
    infer_num_classes_from_checkpoint,
    load_checkpoint,
    make_reconstruction_metadata,
    normalized_model_state,
    reconstruct_model,
    save_checkpoint,
)
from scripts.evaluate import infer_num_classes_from_checkpoint as standalone_classes
from scripts.evaluate_cross_domain import infer_num_classes as cross_classes


def baseline_cfg():
    return {"model": {"name": "reid_baseline", "backbone": {
        "pretrained": False, "last_conv_stride": 1}, "head": {
        "embedding_dim": 8, "bnneck": True, "normalize": True,
        "classifier": True, "eval_feat": "bn"}}}


class Surrogate(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.projection = nn.Linear(4, 6)
        self.heads = nn.ModuleList(nn.Linear(6, num_classes) for _ in range(cfg["model"]["heads"]))
        self.checkpoint_metadata = make_reconstruction_metadata(cfg, num_classes, 6)

    def forward(self, x):
        emb = self.projection(x)
        heads = tuple(head(emb) for head in self.heads)
        return {"emb": emb, "logits": heads[0] if len(heads) == 1 else heads}


@pytest.fixture
def surrogate_builder(monkeypatch):
    calls = []

    def build(cfg, num_classes=None, *, initialize_pretrained=True):
        assert not initialize_pretrained, "Reconstruction attempted pretrained initialization"
        calls.append((copy.deepcopy(cfg), num_classes))
        return Surrogate(cfg, num_classes)

    monkeypatch.setattr(builders, "build_model", build)
    return calls


def payload(tmp_path, heads=6):
    cfg = {"model": {"name": "test_surrogate", "variant": "independent",
                     "heads": heads, "num_classes": 3, "embedding_dim": 6,
                     "backbone": {"pretrained": True}},
           "data": {"train": {"dataset": {"num_classes": 3}}}}
    model = Surrogate(cfg, 3)
    path = tmp_path / "source.pth"
    checkpoint = save_checkpoint(path, model, cfg=cfg, epoch=10)
    return model, checkpoint, path


@pytest.mark.parametrize("heads", [1, 6])
def test_metadata_roundtrip_uses_source_config_and_classes(tmp_path, surrogate_builder, heads):
    model, checkpoint, path = payload(tmp_path, heads)
    target = {"model": {"name": "unrelated", "num_classes": 999},
              "data": {"test": {"dataset": {"num_classes": 999}}}}
    saved_cfg = copy.deepcopy(checkpoint["cfg"])
    loaded = reconstruct_model(torch.load(path, weights_only=True), cfg=target)
    x = torch.randn(2, 4)
    torch.testing.assert_close(loaded(x)["emb"], model(x)["emb"], rtol=0, atol=0)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
    assert all(h.out_features == 3 for h in loaded.heads)
    assert surrogate_builder[-1] == (saved_cfg, 3)
    assert checkpoint["cfg"] == saved_cfg
    assert "cfg" not in checkpoint["reconstruction"]
    for infer in (infer_num_classes_from_checkpoint, standalone_classes, cross_classes):
        assert infer(checkpoint) == 3


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_historical_baseline_reconstructs_without_download(tmp_path, monkeypatch, wrapped, prefix):
    cfg = baseline_cfg()
    original = builders.build_model(cfg, num_classes=3).eval()
    state = original.state_dict()
    if prefix:
        state = {"module." + k: v for k, v in state.items()}
    cfg["model"]["backbone"]["pretrained"] = True
    checkpoint = {"model": state, "cfg": cfg} if wrapped else state
    import reid.models.baseline as baseline
    original_resnet = baseline.resnet50

    def no_download_resnet(*args, **kwargs):
        assert kwargs.get("weights") is None
        return original_resnet(*args, **kwargs)

    monkeypatch.setattr(baseline, "resnet50", no_download_resnet)
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("download"))
    reconstructed = reconstruct_model(checkpoint, cfg=cfg).eval()
    assert cfg["model"]["backbone"]["pretrained"] is True
    x = torch.randn(2, 3, 64, 32)
    with torch.no_grad():
        for name, value in original(x).items():
            torch.testing.assert_close(reconstructed(x)[name], value, rtol=0, atol=0)
    assert standalone_classes(checkpoint) == cross_classes(checkpoint) == 3


def test_new_baseline_metadata_and_no_download(tmp_path, monkeypatch):
    cfg = baseline_cfg()
    model = builders.build_model(cfg, num_classes=3)
    cfg["model"]["backbone"]["pretrained"] = True
    checkpoint = save_checkpoint(tmp_path / "new.pth", model, cfg=cfg)
    assert checkpoint["reconstruction"] == {
        "schema_version": 1, "output_contract_version": 1,
        "model_name": "reid_baseline", "variant": None,
        "num_classes": 3, "embedding_dim": 8,
    }
    from torchvision.models import ResNet50_Weights
    monkeypatch.setattr(type(ResNet50_Weights.IMAGENET1K_V2), "get_state_dict",
                        lambda *a, **k: pytest.fail("pretrained download path invoked"))
    restored = reconstruct_model(checkpoint)
    assert restored.classifier.out_features == 3
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("key,value,match", [
    ("schema_version", 99, "schema_version"),
    ("output_contract_version", 99, "output_contract_version"),
    ("model_name", "wrong", "model_name"),
    ("variant", "wrong", "variant"),
    ("num_classes", 5, "num_classes"),
    ("embedding_dim", 9, "embedding_dim"),
    ("num_classes", True, "positive integer"),
])
def test_bad_metadata_rejected_before_build(tmp_path, surrogate_builder, key, value, match):
    _, checkpoint, _ = payload(tmp_path)
    checkpoint["reconstruction"][key] = value
    with pytest.raises(ValueError, match=match):
        reconstruct_model(checkpoint)
    assert not surrogate_builder


@pytest.mark.parametrize("location", ["model", "head"])
def test_source_config_class_count_mismatch(tmp_path, surrogate_builder, location):
    _, checkpoint, _ = payload(tmp_path)
    section = checkpoint["cfg"]["model"]
    if location == "head":
        section = section.setdefault("head", {})
    section["num_classes"] = 17
    with pytest.raises(ValueError, match="metadata/config mismatch: num_classes"):
        reconstruct_model(checkpoint)


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape"])
def test_strict_tensor_failures(tmp_path, surrogate_builder, kind):
    _, checkpoint, path = payload(tmp_path)
    if kind == "missing":
        del checkpoint["model"]["heads.0.weight"]
    elif kind == "unexpected":
        checkpoint["model"]["extra.weight"] = torch.ones(1)
    else:
        checkpoint["model"]["heads.0.weight"] = torch.ones(9, 6)
    with pytest.raises(RuntimeError, match={"missing": "Missing key", "unexpected": "Unexpected key", "shape": "size mismatch"}[kind]):
        reconstruct_model(checkpoint)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(path, Surrogate(checkpoint["cfg"], 3))


def test_uniform_metadata_prefix_and_mixed_rejection(tmp_path, surrogate_builder):
    model, checkpoint, _ = payload(tmp_path)
    checkpoint["model"] = {"module." + k: v for k, v in model.state_dict().items()}
    assert reconstruct_model(checkpoint).heads[0].out_features == 3
    key = next(iter(checkpoint["model"]))
    checkpoint["model"][key[7:]] = checkpoint["model"].pop(key)
    with pytest.raises(ValueError, match="Mixed module"):
        reconstruct_model(checkpoint)


def test_dataparallel_save_and_state_version_metadata(tmp_path, surrogate_builder):
    model, checkpoint, _ = payload(tmp_path)
    wrapped = nn.DataParallel(model)
    checkpoint = save_checkpoint(tmp_path / "parallel.pth", wrapped, cfg=checkpoint["cfg"])
    state = normalized_model_state(checkpoint)
    assert list(state) == list(model.state_dict())
    assert state._metadata == model.state_dict()._metadata
    reconstruct_model(checkpoint)


@pytest.mark.parametrize("change", ["missing_cfg", "missing_field", "null_metadata", "unknown_legacy"])
def test_no_silent_metadata_fallback(tmp_path, surrogate_builder, change):
    _, checkpoint, _ = payload(tmp_path)
    if change == "missing_cfg":
        del checkpoint["cfg"]
    elif change == "missing_field":
        del checkpoint["reconstruction"]["embedding_dim"]
    elif change == "null_metadata":
        checkpoint["reconstruction"] = None
    else:
        del checkpoint["reconstruction"]
    with pytest.raises(ValueError):
        reconstruct_model(checkpoint)
    assert not surrogate_builder


def test_metadata_disagreement_with_built_model(tmp_path, monkeypatch):
    _, checkpoint, _ = payload(tmp_path)
    def wrong_builder(cfg, num_classes, **kwargs):
        return Surrogate(cfg, num_classes + 1)
    # Remove optional config class declaration so model metadata detects mismatch.
    del checkpoint["cfg"]["model"]["num_classes"]
    monkeypatch.setattr(builders, "build_model", wrong_builder)
    with pytest.raises(ValueError, match="constructed model declaration"):
        reconstruct_model(checkpoint)


def test_legacy_classifier_free_baseline():
    cfg = baseline_cfg()
    cfg["model"]["head"]["classifier"] = False
    model = builders.build_model(cfg)
    restored = reconstruct_model({"model": model.state_dict(), "cfg": cfg})
    assert restored.classifier is None


def test_optional_cfg_save_api_remains_compatible(tmp_path):
    model = builders.build_model(baseline_cfg(), num_classes=3)
    path = tmp_path / "weights_without_config.pth"
    checkpoint = save_checkpoint(path, model)
    assert checkpoint["cfg"] is None
    assert "reconstruction" not in checkpoint
    reconstructed = reconstruct_model(checkpoint, cfg=baseline_cfg())
    assert torch.equal(reconstructed.classifier.weight, model.classifier.weight)


def test_save_rejects_conflicting_config(tmp_path):
    model, checkpoint, _ = payload(tmp_path)
    cfg = copy.deepcopy(checkpoint["cfg"])
    cfg["model"]["num_classes"] = 7
    path = tmp_path / "invalid.pth"
    with pytest.raises(ValueError, match="num_classes"):
        save_checkpoint(path, model, cfg=cfg)
    assert not path.exists()


def test_cross_domain_entry_point_reconstructs_source_model(tmp_path, surrogate_builder, monkeypatch):
    import argparse
    import scripts.evaluate_cross_domain as cross

    model, checkpoint, path = payload(tmp_path)
    source = checkpoint["cfg"]
    source.update({"system": {"device": "cpu"}, "experiment": {}, "data": {
        "root": "source", "num_workers": 0, "pin_memory": False,
        "train": {"dataset": {"name": "source"}},
        "test": {"dataset": {"name": "source"}},
    }})
    torch.save(checkpoint, path)
    target = copy.deepcopy(source)
    target["model"]["num_classes"] = 999
    target["data"]["test"]["dataset"] = {"name": "target", "num_classes": 999}
    args = argparse.Namespace(config="target.yaml", opts=[], checkpoint=str(path),
                              source_dataset="source", target_dataset="target",
                              output_dir=str(tmp_path / "evaluation"))
    monkeypatch.setattr(cross, "parse_args", lambda: args)
    monkeypatch.setattr(cross, "load_config", lambda *a, **k: target)
    monkeypatch.setattr(cross, "validate_config", lambda cfg: None)
    monkeypatch.setattr(cross, "validate_reid_config", lambda cfg: None)
    monkeypatch.setattr(cross, "select_device", lambda *a: (torch.device("cpu"), None))
    monkeypatch.setattr(cross, "build_test_loader", lambda cfg: object())
    def evaluate(cfg, rebuilt, loader, device):
        assert all(h.out_features == 3 for h in rebuilt.heads)
        assert cfg["data"]["test"]["dataset"]["num_classes"] == 999
        assert all(torch.equal(v, rebuilt.state_dict()[k]) for k, v in model.state_dict().items())
        return {"mAP": 0.5}
    monkeypatch.setattr(cross, "evaluate_reid", evaluate)
    monkeypatch.setattr(cross, "build_cross_dataset_record", lambda **kw: kw["scores"])
    cross.main()
    assert surrogate_builder[-1][1] == 3


@pytest.fixture
def real_pcb_checkpoint(tmp_path, monkeypatch):
    import reid.models.pcb as pcb
    def forbidden(*args, **kwargs):
        pytest.fail("Real PCB reconstruction must never initialize/read/download pretrained weights")
    monkeypatch.setattr(pcb, "_read_historical_weights", forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    cfg = {"model": {"name": "pcb", "pretrained": False},
           "data": {"train": {"dataset": {"num_classes": 3}}}}
    torch.manual_seed(42)
    model = builders.build_model(cfg, num_classes=3).eval()
    cfg["model"]["pretrained"] = True
    cfg["model"]["weights_path"] = "must-not-be-opened.pth"
    path = tmp_path / "real-pcb.pth"
    save_checkpoint(path, model, cfg=cfg)
    return model, torch.load(path, map_location="cpu", weights_only=True)


def test_real_pcb_checkpoint_roundtrip_source_classes_state_and_outputs(real_pcb_checkpoint):
    from reid.models.pcb import PCB
    model, checkpoint = real_pcb_checkpoint
    assert "classifier.weight" not in checkpoint["model"]
    assert checkpoint["reconstruction"] == {
        "schema_version": 1, "output_contract_version": 1, "model_name": "pcb",
        "variant": "independent_part_reduction", "num_classes": 3, "embedding_dim": 1536}
    target = {"model": {"name": "pcb", "num_classes": 999},
              "data": {"test": {"dataset": {"num_classes": 999}}}}
    source_cfg_before = copy.deepcopy(checkpoint["cfg"])
    restored = reconstruct_model(checkpoint, cfg=target).eval()
    assert type(restored) is PCB and restored.num_classes == 3
    assert restored.embedding_dim == 1536 and restored.feat_dim is None
    assert all(head.out_features == 3 for head in restored.classifiers.fc_list)
    assert restored.checkpoint_metadata == checkpoint["reconstruction"]
    assert checkpoint["cfg"] == source_cfg_before
    assert list(restored.state_dict()) == list(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        images = torch.randn(1, 3, 384, 128)
        expected, actual = model(images), restored(images)
    assert set(actual) == set(expected)
    assert actual["feat_raw"] is actual["feat_bn"] is None
    torch.testing.assert_close(actual["emb"], expected["emb"], rtol=0, atol=0)
    for value, original in zip(actual["logits"], expected["logits"]):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    for infer in (infer_num_classes_from_checkpoint, standalone_classes, cross_classes):
        assert infer(checkpoint) == 3


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "dimension", "variant"])
def test_real_pcb_checkpoint_rejects_incompatible_state_or_metadata(real_pcb_checkpoint, mutation):
    _, checkpoint = real_pcb_checkpoint
    key = "classifiers.fc_list.0.weight"
    if mutation == "missing":
        del checkpoint["model"][key]
    elif mutation == "unexpected":
        checkpoint["model"]["unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        checkpoint["model"][key] = torch.zeros(99, 256)
    elif mutation == "dimension":
        checkpoint["reconstruction"]["embedding_dim"] = 8
    else:
        checkpoint["reconstruction"]["variant"] = "shared"
    expected = ValueError if mutation in {"dimension", "variant"} else RuntimeError
    with pytest.raises(expected):
        reconstruct_model(checkpoint)
