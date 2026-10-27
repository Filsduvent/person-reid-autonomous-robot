"""Phase 3 CPU contracts only: no datasets, models, training or evaluation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from reid.utils.config import load_config, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.lodo_protocol import (
    DATASET_ORDER, FOLDS, configuration_identity, operational_configuration,
    scientific_configuration, serialize_scientific_configuration, validate_fold,
    validate_lodo_config, validate_output_path,
)


ROOT = Path(__file__).resolve().parents[1]


def descriptor(name, target=False):
    result = dict(name=name, format="raw" if name == "msmt17" else "processed",
                  split="test" if target else ("train" if name == "msmt17" else "trainval"))
    if name == "cuhk03":
        result.update(image_type="detected", protocol="processed_partition", split_id=None,
                      provenance=dict(intended_protocol="detected_new", verification="unverified"))
    if name == "msmt17":
        result.update(lists=["list_query.txt", "list_gallery.txt"] if target else ["list_train.txt"],
                      image_subdir="msmt17/MSMT17_V2/" + ("mask_test_v2" if target else "mask_train_v2"))
    return result


def lodo_config(architecture="resnet50", fold="LODO-1"):
    # Synthetic in-memory extension of existing presets; no executable LODO YAML.
    preset = "configs/pcb/market1501.yaml" if architecture == "pcb" else "configs/baseline_market1501_resnet50_triplet.yaml"
    cfg = load_config(str(ROOT / preset))
    cfg["protocol"] = dict(name="multi_source_lodo", version=1, architecture=architecture, fold=fold)
    sources, target = FOLDS[fold]
    cfg["data"]["train"]["dataset"] = dict(name="multi_source", sources=[descriptor(s) for s in sources])
    cfg["data"]["test"]["dataset"] = descriptor(target, target=True)
    cfg["data"]["root"] = "/synthetic/not-opened"
    cfg["experiment"] = dict(name="synthetic_lodo", output_dir=f"exp/lodo/v1/{architecture}/{fold}/seed42/run-1")
    cfg["train"] = dict(mode="successful_updates", max_updates=94200 if architecture == "pcb" else 85200,
                        evaluate_during_training=False, checkpoint_selection="terminal",
                        save=dict(save_best=False, save_last=True, save_terminal=True, resume=""))
    cfg["sched"].update(unit="successful_primary_updates", milestones=[31400] if architecture == "pcb" else [28400, 49700])
    if architecture == "pcb":
        cfg["data"]["train"]["batch"]["drop_last"] = True
    else:
        cfg["loss"]["center"].update(enabled=True, feat="raw", reduction="half_mean_squared_distance")
    return cfg


def set_value(cfg, path, value):
    node = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


@pytest.mark.parametrize("fold,target", list(zip(FOLDS, DATASET_ORDER)))
def test_four_fold_definitions(fold, target):
    sources, actual_target = FOLDS[fold]
    assert actual_target == target
    assert sources == tuple(d for d in DATASET_ORDER if d != target)
    validate_fold(fold, list(sources), target)


@pytest.mark.parametrize("fold,sources,target", [
    ("unknown", ["duke", "cuhk03", "msmt17"], "market1501"),
    ("LODO-1", ["duke", "cuhk03"], "market1501"),
    ("LODO-1", ["duke", "duke", "msmt17"], "market1501"),
    ("LODO-1", ["market1501", "cuhk03", "msmt17"], "market1501"),
    ("LODO-1", ["cuhk03", "duke", "msmt17"], "market1501"),
    ("LODO-1", ["duke", "unknown", "msmt17"], "market1501"),
    ("LODO-1", ["duke", "cuhk03", "msmt17"], "duke"),
    ("LODO-1", None, "market1501"),
])
def test_invalid_folds(fold, sources, target):
    with pytest.raises(ValueError):
        validate_fold(fold, sources, target)


@pytest.mark.parametrize("architecture", ["resnet50", "pcb"])
@pytest.mark.parametrize("fold", list(FOLDS))
def test_valid_architecture_fold_configs(architecture, fold, tmp_path):
    cfg = lodo_config(architecture, fold)
    before = copy.deepcopy(cfg)
    validate_config(cfg, project_root=tmp_path)
    validate_reid_config(cfg, num_classes=2510, project_root=tmp_path)
    assert cfg == before
    assert not (tmp_path / "exp").exists()
    identity = scientific_configuration(cfg)
    assert identity["architecture_contract"] == dict(
        embedding_dim=1536 if architecture == "pcb" else 2048,
        classifier_heads=6 if architecture == "pcb" else 1)


@pytest.mark.parametrize("architecture", ["resnet50", "pcb"])
@pytest.mark.parametrize("path,value", [
    ("protocol.name", "lodo"), ("protocol.version", 2), ("protocol.version", True),
    ("protocol.architecture", "mgn"), ("protocol.fold", "LODO-5"),
    ("train.mode", "epochs"), ("train.epochs", 120), ("train.epochs", None),
    ("train.eval_interval", 10), ("train.max_updates", 120),
    ("train.evaluate_during_training", True), ("train.evaluate_during_training", 0),
    ("train.checkpoint_selection", "best"), ("train.save.save_best", True),
    ("train.save.save_terminal", False), ("train.save.overwrite", True),
    ("system.amp", True), ("system.amp", "false"), ("system.precision", "fp16"),
    ("system.device", "gpu"), ("repro.seed", 1337),
    ("optim.weight_decay", 0.0), ("optim.bias_lr_factor", 2.0),
    ("optim.weight_decay_bias", 0.0), ("optim.betas", [0.5, 0.9]),
    ("sched.unit", "epochs"), ("sched.milestones", [40, 70]),
    ("sched.gamma", 0.2), ("sched.warmup_method", "constant"),
    ("sched.warmup_denominator", 499), ("data.train.batch.batch_size", 32),
    ("data.train.batch.domain_balanced", True), ("data.train.images.size", [128, 256]),
    ("data.test.images.size", [128, 256]), ("data.test.loader.shuffle", True),
    ("loss.id.enabled", False), ("loss.id.weight", 0.5), ("loss.id.weight", True),
    ("eval.normalize_feat", False), ("eval.rerank.enabled", True),
])
def test_common_contract_rejections(architecture, path, value, tmp_path):
    cfg = lodo_config(architecture)
    set_value(cfg, path, value)
    for validator in (validate_config, validate_reid_config):
        with pytest.raises(ValueError):
            validator(cfg, project_root=tmp_path)


@pytest.mark.parametrize("path,value", [
    ("model.backbone.last_conv_stride", 2), ("model.head.bnneck", False),
    ("model.head.embedding_dim", 1024), ("model.head.metric_feat", "bn"),
    ("model.head.eval_feat", "raw"), ("loss.id.label_smoothing", 0),
    ("loss.triplet.enabled", False), ("loss.triplet.weight", 2.0),
    ("loss.triplet.margin", 0.2), ("loss.center.enabled", False),
    ("loss.center.weight", 0.001), ("loss.center.lr", 1.0),
    ("loss.center.feat", "bn"), ("loss.center.reduction", "mean_squared_distance"),
    ("loss.center.momentum", 0.9), ("data.train.batch.P", 8),
    ("data.train.batch.K", 8), ("data.train.batch.sampler", "random"),
    ("optim.name", "sgd"), ("optim.lr", 0.00035),
    ("sched.warmup_iters", 499), ("sched.warmup_factor", 0.1),
    ("sched.milestones", [28400.0, 49700]), ("train.max_updates", 85200.0),
])
def test_resnet_recipe_rejections(path, value, tmp_path):
    cfg = lodo_config()
    set_value(cfg, path, value)
    with pytest.raises(ValueError):
        validate_config(cfg, project_root=tmp_path)


@pytest.mark.parametrize("path,value", [
    ("model.variant", "shared"), ("model.embedding_dim", 1536), ("model.num_heads", 6),
    ("loss.id.head_aggregation", "mean"), ("loss.id.label_smoothing", 0.1),
    ("loss.triplet.enabled", True), ("loss.center.enabled", True),
    ("data.train.batch.sampler", "pk"), ("data.train.batch.drop_last", False),
    ("optim.name", "adam"), ("optim.lr", 0.01), ("optim.momentum", 0),
    ("optim.nesterov", True), ("optim.param_groups", []),
    ("optim.param_groups", [{"prefix": "backbone.", "lr_mult": 1.0}]),
    ("sched.warmup_iters", 500), ("sched.milestones", [31401]),
])
def test_pcb_recipe_rejections(path, value, tmp_path):
    cfg = lodo_config("pcb")
    set_value(cfg, path, value)
    with pytest.raises(ValueError):
        validate_config(cfg, project_root=tmp_path)


@pytest.mark.parametrize("index,key,value", [
    (0, "format", "raw"), (0, "split", "test"), (0, "name", "combined"),
    (1, "image_type", "labeled"), (1, "protocol", "new"), (1, "split_id", 0),
    (1, "provenance", {"intended_protocol": "detected_new", "verification": "verified"}),
    (2, "format", "processed"), (2, "split", "trainval"),
    (2, "lists", ["list_train.txt", "list_val.txt"]),
    (2, "image_subdir", "msmt17/MSMT17_V2/mask_test_v2"),
    (0, "metadata_sha256", "invalid"), (0, "unknown_option", True),
])
def test_dataset_descriptor_rejections(index, key, value, tmp_path):
    cfg = lodo_config()
    cfg["data"]["train"]["dataset"]["sources"][index][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, project_root=tmp_path)


@pytest.mark.parametrize("mutation", ["wrong_target", "target_split", "source_target", "duplicate", "order", "missing"])
def test_descriptor_fold_validation(mutation, tmp_path):
    cfg = lodo_config()
    sources = cfg["data"]["train"]["dataset"]["sources"]
    if mutation == "wrong_target": cfg["data"]["test"]["dataset"] = descriptor("duke", True)
    elif mutation == "target_split": cfg["data"]["test"]["dataset"]["split"] = "trainval"
    elif mutation == "source_target": sources[0] = descriptor("market1501")
    elif mutation == "duplicate": sources[1] = copy.deepcopy(sources[0])
    elif mutation == "order": sources.reverse()
    else: sources.pop()
    with pytest.raises(ValueError): validate_config(cfg, project_root=tmp_path)


def test_lodo_cannot_fall_back_to_stage_a(tmp_path):
    cfg = lodo_config()
    del cfg["protocol"]
    cfg["train"]["epochs"] = 120
    for validator in (validate_config, validate_reid_config):
        with pytest.raises(ValueError, match="explicit protocol"):
            validator(cfg, project_root=tmp_path)
    cfg["protocol"] = None
    with pytest.raises(ValueError): validate_config(cfg, project_root=tmp_path)


@pytest.mark.parametrize("path", ["protocol.version", "train.mode", "train.max_updates", "train.checkpoint_selection",
                                 "train.save.save_terminal", "sched.unit", "loss.center.reduction"])
def test_required_stage_b_fields(path, tmp_path):
    cfg = lodo_config()
    node = cfg
    parts = path.split(".")
    for key in parts[:-1]: node = node[key]
    del node[parts[-1]]
    with pytest.raises(ValueError): validate_config(cfg, project_root=tmp_path)


def test_identity_stability_and_operational_separation():
    cfg = lodo_config()
    changed = copy.deepcopy(cfg)
    changed["experiment"].update(name="different", output_dir="exp/lodo/v1/resnet50/LODO-1/seed42/run-2", notes="note")
    changed["train"]["save"]["resume"] = "/temporary/checkpoint.pth"
    changed["data"].update(root="/another/host/root", num_workers=0, pin_memory=False)
    changed["data"]["train"]["dataset"]["sources"][0]["root"] = "/temporary/source"
    changed["system"].update(device="cpu", gpu_id=7)
    changed["logging"] = {"to_file": False}
    changed["runtime"] = {"timestamp": "later", "duration": 30}
    changed["loss"]["id"]["weight"] = 1
    changed = json.loads(json.dumps(changed, sort_keys=True))
    assert configuration_identity(cfg) == configuration_identity(changed)
    assert operational_configuration(cfg) != operational_configuration(changed)
    raw = serialize_scientific_configuration(cfg)
    assert configuration_identity(cfg) == hashlib.sha256(raw).hexdigest()
    assert raw == json.dumps(scientific_configuration(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    assert len(configuration_identity(cfg)) == 64
    for other in [lodo_config(fold="LODO-2"), lodo_config("pcb")]:
        assert configuration_identity(other) != configuration_identity(cfg)
    changed = copy.deepcopy(cfg)
    changed["data"]["train"]["dataset"]["sources"][0]["metadata_sha256"] = "a" * 64
    assert configuration_identity(changed) != configuration_identity(cfg)
    before = copy.deepcopy(cfg)
    scientific_configuration(cfg)["loss"]["id"]["weight"] = 5
    assert cfg == before


def test_equivalent_inferred_descriptors_have_same_identity():
    cfg = lodo_config()
    other = copy.deepcopy(cfg)
    del other["data"]["train"]["dataset"]["sources"][1]["provenance"]
    del other["data"]["train"]["dataset"]["sources"][2]["lists"]
    del other["data"]["train"]["dataset"]["sources"][2]["image_subdir"]
    assert configuration_identity(cfg) == configuration_identity(other)


@pytest.mark.parametrize("value", ["", " ", "exp/pcb/market1501", "exp/market1501_full_bag_of_tricks",
    "../escape", "exp/lodo/v1/resnet50/LODO-1/seed42/../escape",
    "exp/lodo/v1/pcb/LODO-1/seed42/run-1", "exp/lodo/v1/resnet50/LODO-2/seed42/run-1",
    "exp/lodo/v1/resnet50/LODO-1/seed1337/run-1", "exp/lodo/v1/resnet50/LODO-1/seed42/<run-id>",
    "exp/lodo/v1/resnet50/LODO-1/seed42/run-1/extra"])
def test_invalid_output_namespaces(value, tmp_path):
    cfg = lodo_config(); cfg["experiment"]["output_dir"] = value
    with pytest.raises(ValueError): validate_output_path(cfg, project_root=tmp_path)


def test_absolute_output_is_valid_only_under_authorized_root(tmp_path):
    cfg = lodo_config()
    intended = tmp_path / cfg["experiment"]["output_dir"]
    cfg["experiment"]["output_dir"] = str(intended)
    assert validate_output_path(cfg, project_root=tmp_path) == intended
    cfg["experiment"]["output_dir"] = str(tmp_path.parent / "exp/lodo/v1/resnet50/LODO-1/seed42/run-1")
    with pytest.raises(ValueError): validate_output_path(cfg, project_root=tmp_path)
    assert not (tmp_path / "exp").exists()


@pytest.mark.parametrize("collision", ["existing", "existing_file", "symlink_stage_a", "symlink_lodo", "symlink_escape", "parent_symlink", "exp_symlink", "parent_file"])
def test_output_collisions_and_symlink_protection(collision, tmp_path):
    cfg = lodo_config()
    output = tmp_path / cfg["experiment"]["output_dir"]
    if collision == "exp_symlink":
        outside = tmp_path / "stage_a_outputs"; outside.mkdir()
        (tmp_path / "exp").symlink_to(outside, target_is_directory=True)
    elif collision == "parent_file":
        (tmp_path / "exp").write_text("file")
    elif collision == "parent_symlink":
        outside = tmp_path / "unrelated"; outside.mkdir()
        output.parent.parent.mkdir(parents=True)
        output.parent.symlink_to(outside, target_is_directory=True)
    else:
        output.parent.mkdir(parents=True)
        if collision == "existing": output.mkdir()
        elif collision == "existing_file": output.write_text("unrelated")
        else:
            destination = (tmp_path / "exp/pcb/market1501" if collision == "symlink_stage_a"
                           else output.parent / "other-run" if collision == "symlink_lodo"
                           else tmp_path / "outside")
            destination.mkdir(parents=True)
            output.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError): validate_config(cfg, project_root=tmp_path)


def test_schema_never_accesses_datasets(tmp_path, monkeypatch):
    # Only output namespace inspection is authorized at this stage.
    original = Path.exists
    def guarded(path):
        assert not str(path).startswith("/synthetic")
        return original(path)
    monkeypatch.setattr(Path, "exists", guarded)
    validate_lodo_config(lodo_config(), project_root=tmp_path)


LEGACY_CONFIG_ERRORS = {
    "configs/baseline_cuhk03_resnet_triplet.yaml": "Missing config section: repro",
    "configs/baseline_r50_triplet.yml": "Missing config section: experiment",
}
STAGE_A_CONFIGS = sorted(set(ROOT.glob("configs/baseline*.yaml")) |
                         set(ROOT.glob("configs/baseline*.yml")) |
                         set(ROOT.glob("configs/experiment*.yaml")) |
                         set(ROOT.glob("configs/pcb/*.yaml")))


@pytest.mark.parametrize("path", STAGE_A_CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_existing_stage_a_config_acceptance_is_preserved(path):
    cfg = load_config(str(path)); before = copy.deepcopy(cfg)
    legacy_error = LEGACY_CONFIG_ERRORS.get(str(path.relative_to(ROOT)))
    if legacy_error:
        # Both failures were verified against the unmodified base schema.
        with pytest.raises(ValueError) as error:
            validate_config(cfg)
        assert str(error.value) == legacy_error
    else:
        validate_config(cfg); validate_reid_config(cfg)
        assert "epochs" in cfg["train"] and "max_updates" not in cfg["train"]
    assert cfg == before and "protocol" not in cfg


def test_stage_a_still_requires_epochs():
    cfg = load_config(str(ROOT / "configs/pcb/market1501.yaml"))
    del cfg["train"]["epochs"]
    with pytest.raises(ValueError, match="train.epochs"):
        validate_config(cfg)


@pytest.mark.parametrize('architecture,expected', [
    ('resnet50', ['a681c90890ebf0006f8709d67ff8c0d246d1f66a2365db53060287f87e69db1b',
                 '35100c2ff4e74d501393fc55b951318be0767947bb55b211f2dac86958874762',
                 'fbeede6c9b35ebcc49d61f49de00fe6bb8eb54b6d0822d3d829c398fa51091a1',
                 '56f38a33fbddf77fb5e326d699f3c26516f7a3cdaaad0ceef628c0b246de6010']),
    ('pcb', ['419b74cbe1217848a78df18959f83d543a79895155356a6e9d0b9df9edcd1f23',
             '88dae2a85e1d9e572de0456fc122f2cdaf5cbd480898fa6a810a92b4dfc5ae96',
             'b2b33be146001c1d0306fe6892620803e50559416f4d940cda392614f8f810b7',
             'c44ae77ecf8807b70457a7a930b454788a977700a89a9b874bc2c27302966d6b'])])
def test_mgn_extensions_preserve_all_eight_historical_scientific_identities(architecture, expected):
    # Captured against be628bbb's original pure scientific identity functions.
    for index, digest in enumerate(expected, 1):
        cfg = lodo_config(architecture, f'LODO-{index}')
        assert configuration_identity(cfg) == digest
        cfg['eval']['flip_aggregation'] = 'none'
        for role in ['train', 'test']:
            cfg['data'][role]['images']['interpolation'] = 'bilinear'
        assert configuration_identity(cfg) == digest


@pytest.mark.parametrize('architecture', ['resnet50', 'pcb'])
def test_lodo_v1_rejects_execution_changing_flip_or_interpolation(architecture):
    cfg = lodo_config(architecture)
    cfg['eval']['flip_aggregation'] = 'mean'
    with pytest.raises(ValueError, match='flip_aggregation'):
        validate_lodo_config(cfg)
    cfg['eval']['flip_aggregation'] = 'none'
    for role in ['train', 'test']:
        changed = copy.deepcopy(cfg)
        changed['data'][role]['images']['interpolation'] = 'bicubic'
        with pytest.raises(ValueError, match='interpolation'):
            validate_lodo_config(changed)
    cfg['protocol']['architecture'] = 'mgn'
    with pytest.raises(ValueError, match='architecture'):
        validate_lodo_config(cfg)
