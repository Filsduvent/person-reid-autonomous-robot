"""Pure configuration contracts for lodo_v1; no data or training execution.

Scientific identity v1 uses canonical UTF-8 JSON (sorted keys, compact
separators, no NaN), with integer-valued floats normalized to integers.
Ordered sources are significant. Dataset content hashes, when supplied, are
scientific; host roots, run IDs, resume paths, devices and logging are
operational. The identity includes the logical output namespace, not its
host-specific root. It is not a complete dataset/run manifest.

Output validation is read-only and reserves fresh paths only. Existing run reuse requires a verified same-run Stage B checkpoint. Checking a path does not atomically reserve it for a future launch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path


PROTOCOL_ID = "multi_source_lodo"
PROTOCOL_VERSION = 1
SCIENTIFIC_PROTOCOL = "lodo_v1"
IDENTITY_VERSION = 1
DATASET_ORDER = ("market1501", "duke", "cuhk03", "msmt17")
ARCHITECTURES = ("resnet50", "pcb")
FOLDS = {
    f"LODO-{index + 1}": (tuple(d for d in DATASET_ORDER if d != target), target)
    for index, target in enumerate(DATASET_ORDER)
}


def has_lodo_protocol(cfg):
    """Presence opts in; malformed or unknown protocol values never fall back."""
    return "protocol" in cfg


def require_explicit_lodo(cfg):
    if has_lodo_protocol(cfg):
        return
    train = cfg.get("train", {})
    data = cfg.get("data", {}).get("train", {})
    data = data.get("dataset", {}) if isinstance(data, dict) else {}
    data = data if isinstance(data, dict) else {}
    if (train.get("mode") == "successful_updates" or "max_updates" in train
            or data.get("name") == "multi_source" or "sources" in data
            or cfg.get("sched", {}).get("unit") == "successful_primary_updates"
            or "cuda_policy" in cfg.get("repro", {})):
        raise ValueError("LODO fields require explicit protocol opt-in")


def _mapping(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _at(cfg, path):
    current = cfg
    for part in path.split("."):
        current = _mapping(current, path)
        if part not in current:
            raise ValueError(f"Missing LODO config key: {path}")
        current = current[part]
    return current


def _equal(actual, expected):
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _equal(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isfinite(actual) and actual == expected
    return actual == expected


def _expect(cfg, path, expected):
    actual = _at(cfg, path)
    if not _equal(actual, expected):
        raise ValueError(f"{path} must be {expected!r}, got {actual!r}")


def _keys(value, allowed, path):
    value = _mapping(value, path)
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    if set(value) - set(allowed):
        raise ValueError(f"Unsupported {path} fields: {sorted(set(value) - set(allowed))}")


def validate_fold(fold, sources, target):
    if not isinstance(fold, str) or fold not in FOLDS:
        raise ValueError(f"Unknown LODO fold: {fold!r}")
    if not isinstance(sources, (list, tuple)) or any(not isinstance(s, str) for s in sources):
        raise ValueError("LODO sources must be ordered dataset keys")
    if not isinstance(target, str) or target not in DATASET_ORDER or any(s not in DATASET_ORDER for s in sources):
        raise ValueError("Unknown LODO dataset key")
    if len(sources) != 3 or len(set(sources)) != 3:
        raise ValueError("LODO requires three distinct sources")
    if target in sources:
        raise ValueError("Held-out target cannot be a training source")
    expected_sources, expected_target = FOLDS[fold]
    if target != expected_target or tuple(sources) != expected_sources:
        raise ValueError("LODO fold complement, target or canonical source order is incorrect")


def _descriptor(value, role):
    value = _mapping(value, f"{role} descriptor")
    _keys(value, {"name", "format", "split", "root", "image_type", "protocol",
                  "split_id", "provenance", "lists", "image_subdir", "metadata_sha256"}, role)
    name = value.get("name")
    if name not in DATASET_ORDER:
        raise ValueError(f"Unknown {role} dataset key: {name!r}")
    expected = {"name": name, "format": "raw" if name == "msmt17" else "processed",
                "split": "test" if role == "target" else ("train" if name == "msmt17" else "trainval")}
    if name == "cuhk03":
        expected.update(image_type="detected", protocol="processed_partition", split_id=None)
    for key, item in expected.items():
        if key not in value or not _equal(value[key], item):
            raise ValueError(f"Unauthorized {role} descriptor {name}.{key}; expected {item!r}")
    specific = {"image_type", "protocol", "split_id", "provenance"}
    if name != "cuhk03" and specific.intersection(value):
        raise ValueError("CUHK03 fields cannot be applied to another dataset")
    if name == "cuhk03":
        provenance = value.get("provenance", {"intended_protocol": "detected_new", "verification": "unverified"})
        if provenance != {"intended_protocol": "detected_new", "verification": "unverified"}:
            raise ValueError("CUHK03 official-new provenance remains unverified")
    if name == "msmt17":
        lists = ["list_train.txt"] if role == "source" else ["list_query.txt", "list_gallery.txt"]
        folder = "mask_train_v2" if role == "source" else "mask_test_v2"
        if "lists" in value and value["lists"] != lists:
            raise ValueError("Unauthorized MSMT17 lists")
        if "image_subdir" in value and value["image_subdir"] != f"msmt17/MSMT17_V2/{folder}":
            raise ValueError("Unauthorized MSMT17 image subtree")
    elif {"lists", "image_subdir"}.intersection(value):
        raise ValueError("List/subtree selectors are MSMT17-only")
    if "root" in value and (not isinstance(value["root"], str) or not value["root"].strip()):
        raise ValueError("Dataset root must be a nonempty string")
    if "metadata_sha256" in value and (not isinstance(value["metadata_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["metadata_sha256"]) is None):
        raise ValueError("Dataset metadata_sha256 must be lowercase SHA256")
    normalized = {k: copy.deepcopy(v) for k, v in value.items() if k != "root"}
    if name == "cuhk03":
        normalized["provenance"] = copy.deepcopy(provenance)
    if name == "msmt17":
        normalized.update(lists=lists, image_subdir=f"msmt17/MSMT17_V2/{folder}")
    return normalized


def _contract(cfg):
    if cfg.get('eval', {}).get('flip_aggregation', 'none') != 'none':
        raise ValueError('LODO v1 requires flip_aggregation=none; historical retrieval identity is frozen.')
    protocol = _at(cfg, "protocol")
    _keys(protocol, {"name", "version", "architecture", "fold"}, "protocol")
    _expect(cfg, "protocol.name", PROTOCOL_ID)
    _expect(cfg, "protocol.version", PROTOCOL_VERSION)
    _keys(_at(cfg, "system"), {"device", "gpu_id", "amp", "log_interval"}, "system")
    _keys(_at(cfg, "repro"), {"seed", "deterministic", "benchmark", "cuda_policy"}, "repro")
    if _at(cfg, "system.device") not in ("auto", "cpu", "cuda"):
        raise ValueError("Unsupported LODO system.device")
    architecture = _at(cfg, "protocol.architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError("Unsupported LODO architecture")
    _expect(cfg, "data.train.dataset.name", "multi_source")
    _keys(_at(cfg, "data.train.dataset"), {"name", "sources"}, "data.train.dataset")
    descriptors = _at(cfg, "data.train.dataset.sources")
    if not isinstance(descriptors, list):
        raise ValueError("LODO sources must be a list of descriptors")
    sources = [_descriptor(s, "source") for s in descriptors]
    target = _descriptor(_at(cfg, "data.test.dataset"), "target")
    validate_fold(_at(cfg, "protocol.fold"), [s["name"] for s in sources], target["name"])
    train = _at(cfg, "train")
    _keys(train, {"mode", "max_updates", "evaluate_during_training", "checkpoint_selection", "save", "recovery"}, "train")
    from reid.utils.lodo_recovery import recovery_policy
    recovery_policy(cfg)
    common = {
        "train.mode": "successful_updates", "train.evaluate_during_training": False,
        "train.checkpoint_selection": "terminal", "train.save.save_best": False,
        "train.save.save_last": True, "train.save.save_terminal": True,
        "system.amp": False, "repro.seed": 42,
        "sched.name": "warmup_multistep", "sched.unit": "successful_primary_updates",
        "sched.gamma": 0.1, "sched.warmup_method": "linear",
        "data.train.batch.batch_size": 64, "loss.id.enabled": True, "loss.id.weight": 1.0,
        "optim.weight_decay": 0.0005, "optim.bias_lr_factor": 1.0,
        "optim.weight_decay_bias": 0.0005,
        "eval.normalize_feat": True, "eval.distance": "euclidean", "eval.topk": [1, 5, 10],
        "eval.rerank.enabled": False, "data.test.loader.shuffle": False,
    }
    for path, expected in common.items():
        _expect(cfg, path, expected)
    _keys(_at(cfg, "train.save"), {"save_best", "save_last", "save_terminal", "resume"}, "train.save")
    _keys(_at(cfg, "sched"), {"name", "unit", "milestones", "gamma", "warmup_iters", "warmup_factor", "warmup_method"}, "sched")
    if not isinstance(_at(cfg, "data.root"), str) or not _at(cfg, "data.root").strip():
        raise ValueError("data.root must be a nonempty string")
    if "resume" in train["save"] and not isinstance(train["save"]["resume"], str):
        raise ValueError("train.save.resume must be a string")
    _keys(_at(cfg, "loss"), {"id", "triplet", "center"}, "loss")
    _keys(_at(cfg, "loss.id"), {"enabled", "weight", "label_smoothing", "head_aggregation"}, "loss.id")
    _keys(_at(cfg, "loss.triplet"), {"enabled", "weight", "margin", "mining"}, "loss.triplet")
    _keys(_at(cfg, "loss.center"), {"enabled", "weight", "lr", "feat", "reduction"}, "loss.center")
    _keys(_at(cfg, "optim"), {"name", "lr", "weight_decay", "bias_lr_factor", "weight_decay_bias", "momentum", "nesterov", "param_groups"}, "optim")
    from reid.utils.lodo_cuda_policy import policy_name, STRICT_POLICY
    strict = policy_name(cfg) == STRICT_POLICY
    for key, expected in {"deterministic": strict, "benchmark": not strict}.items():
        if key in cfg["repro"]:
            _expect(cfg, f"repro.{key}", expected)
    if architecture == "resnet50":
        recipe = {
            "model.name": "reid_baseline", "model.backbone.name": "resnet50",
            "model.backbone.pretrained": True, "model.backbone.last_conv_stride": 1,
            "model.head.embedding_dim": 2048, "model.head.pooling": "gap",
            "model.head.bnneck": True, "model.head.normalize": True,
            "model.head.metric_feat": "raw", "model.head.eval_feat": "bn",
            "model.head.classifier": True, "loss.id.label_smoothing": 0.1,
            "loss.triplet.enabled": True, "loss.triplet.weight": 1.0,
            "loss.triplet.margin": 0.3, "loss.triplet.mining": "batch_hard",
            "loss.center.enabled": True, "loss.center.weight": 0.0005,
            "loss.center.lr": 0.5, "loss.center.feat": "raw",
            "loss.center.reduction": "half_mean_squared_distance",
            "data.train.batch.sampler": "pk", "data.train.batch.P": 16, "data.train.batch.K": 4,
            "optim.name": "adam", "optim.lr": 0.0003,
            "train.max_updates": 85200, "sched.milestones": [28400, 49700],
            "sched.warmup_iters": 500, "sched.warmup_factor": 0.01,
        }
        _keys(_at(cfg, "model"), {"name", "backbone", "head"}, "model")
        _keys(_at(cfg, "model.backbone"), {"name", "pretrained", "last_conv_stride"}, "model.backbone")
        _keys(_at(cfg, "model.head"), {"embedding_dim", "pooling", "bnneck", "normalize", "metric_feat", "eval_feat", "classifier"}, "model.head")
        if cfg["optim"].get("param_groups", []) != []:
            raise ValueError("ResNet LODO has no optimizer prefix overrides")
        for key, expected in {"momentum": 0.9, "nesterov": False}.items():
            if key in cfg["optim"]:
                _expect(cfg, f"optim.{key}", expected)
        batch_keys = {"sampler", "P", "K", "batch_size", "drop_last"}
        if "drop_last" in cfg["data"]["train"]["batch"]:
            _expect(cfg, "data.train.batch.drop_last", True)
    else:
        recipe = {
            "model.name": "pcb", "model.variant": "independent_part_reduction",
            "model.pretrained": True, "loss.id.label_smoothing": 0.0,
            "loss.id.head_aggregation": "sum", "loss.triplet.enabled": False,
            "loss.center.enabled": False, "data.train.batch.sampler": "random",
            "data.train.batch.drop_last": True, "optim.name": "sgd", "optim.lr": 0.1,
            "optim.momentum": 0.9, "optim.nesterov": False,
            "optim.param_groups": [{"prefix": "backbone.", "lr_mult": 0.1}],
            "train.max_updates": 94200, "sched.milestones": [31400],
            "sched.warmup_iters": 0, "sched.warmup_factor": 1.0,
        }
        _keys(_at(cfg, "model"), {"name", "variant", "pretrained", "weights_path"}, "model")
        batch_keys = {"sampler", "batch_size", "drop_last"}
        _keys(_at(cfg, "loss.triplet"), {"enabled"}, "loss.triplet")
        _keys(_at(cfg, "loss.center"), {"enabled"}, "loss.center")
    _keys(_at(cfg, "data.train.batch"), batch_keys, "data.train.batch")
    if "head_aggregation" in cfg["loss"]["id"]:
        _expect(cfg, "loss.id.head_aggregation", "sum")
    for path, expected in recipe.items():
        _expect(cfg, path, expected)
    size = [256, 128] if architecture == "resnet50" else [384, 128]
    for role in ("train", "test"):
        _expect(cfg, f"data.{role}.images.size", size)
        images = _at(cfg, f"data.{role}.images")
        if images.get('interpolation') not in (None, 'bilinear'):
            raise ValueError('LODO v1 requires historical bilinear interpolation.')
        if images.get("dir") is not None or images.get("batch_dims", "NCHW") != "NCHW":
            raise ValueError("LODO uses descriptor-owned images in NCHW layout")
        for key, expected in {"mean": [0.486, 0.459, 0.408], "std": [0.229, 0.224, 0.225],
                              "mirror": "random" if role == "train" else "none"}.items():
            _expect(cfg, f"data.{role}.aug.{key}", expected)
    if cfg["data"]["train"].get("loader", {}).get("shuffle", True) is not True:
        raise ValueError("LODO training must retain shuffled sampling")
    if architecture == "resnet50":
        for key, expected in {"padding.enabled": True, "padding.size": 10,
                              "random_crop.enabled": True, "random_erasing.enabled": True,
                              "random_erasing.probability": 0.5}.items():
            _expect(cfg, f"data.train.aug.{key}", expected)
    else:
        for key in ("padding", "random_crop", "random_erasing"):
            _expect(cfg, f"data.train.aug.{key}.enabled", False)
    return sources, target


def validate_output_path(cfg, *, project_root=None):
    """Check a fresh namespace without making directories or reading datasets."""
    project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    experiment = _at(cfg, "experiment")
    _keys(experiment, {"name", "output_dir", "notes"}, "experiment")
    value = _at(cfg, "experiment.output_dir")
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
        raise ValueError("Invalid LODO output path")
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("LODO output path traversal is prohibited")
    lexical = path if path.is_absolute() else project / path
    root = project / "exp"
    expected = ("lodo", "v1", _at(cfg, "protocol.architecture"), _at(cfg, "protocol.fold"), "seed42")
    try:
        parts = lexical.relative_to(root).parts
        if len(parts) != 6 or parts[:5] != expected or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parts[-1]) is None:
            raise ValueError("Output must use exp/lodo/v1/<architecture>/<fold>/seed42/<run_id>")
        resolved = lexical.resolve()
        if root.resolve() != root or resolved.relative_to(root).parts != parts:
            raise ValueError("LODO output symlink alias or escape is prohibited")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unsafe LODO output namespace: {value!r}: {error}") from error
    if lexical.exists() or lexical.is_symlink():
        if not lexical.is_dir() or not cfg['train']['save'].get('resume'):
            raise ValueError("Existing LODO output requires verified compatible resume")
        from reid.utils.lodo_checkpoint import validate_resume_path
        validate_resume_path(cfg, resolved)
    if any(parent.exists() and not parent.is_dir() for parent in lexical.parents if parent != project.parent):
        raise ValueError("LODO output parent is not a directory")
    return resolved


def validate_lodo_config(cfg, *, project_root=None):
    _contract(cfg)
    validate_output_path(cfg, project_root=project_root)


def _canonical_numbers(value):
    if isinstance(value, dict):
        return {k: _canonical_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical_numbers(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def scientific_configuration(cfg):
    """Return immutable scientific metadata; no output reservation/manifest IO."""
    sources, target = _contract(cfg)
    architecture = cfg["protocol"]["architecture"]
    model = {k: copy.deepcopy(v) for k, v in cfg["model"].items() if k != "weights_path"}
    optimizer = copy.deepcopy(cfg["optim"])
    if architecture == "resnet50":
        for key in ("momentum", "nesterov", "param_groups"):
            optimizer.pop(key, None)  # These optional default fields do not alter Adam.
    from reid.utils.lodo_cuda_policy import policy_contract
    execution = policy_contract(cfg)
    return _canonical_numbers({
        **({"execution_policy": execution} if execution is not None else {}),
        "identity_schema_version": 2 if execution is not None else IDENTITY_VERSION,
        "scientific_protocol": SCIENTIFIC_PROTOCOL, "protocol": copy.deepcopy(cfg["protocol"]),
        "sources": sources, "target": target, "seed": cfg["repro"]["seed"],
        "model": model, "loss": copy.deepcopy(cfg["loss"]), "optim": optimizer,
        "sampling": copy.deepcopy(cfg["data"]["train"]["batch"]),
        "transforms": {role: {"size": cfg["data"][role]["images"]["size"],
                              "aug": copy.deepcopy(cfg["data"][role]["aug"])} for role in ("train", "test")},
        "update_budget": cfg["train"]["max_updates"], "scheduler": copy.deepcopy(cfg["sched"]),
        "checkpoint_selection": "terminal", "evaluate_during_training": False,
        "precision": "fp32", "deterministic": cfg["repro"].get("deterministic", False),
        "benchmark": cfg["repro"].get("benchmark", True),
        "retrieval": {key: copy.deepcopy(cfg["eval"][key]) for key in ("normalize_feat", "distance", "topk", "rerank")},
        "architecture_contract": {"embedding_dim": 2048 if architecture == "resnet50" else 1536,
                                  "classifier_heads": 1 if architecture == "resnet50" else 6},
        "output_namespace": f"exp/lodo/v1/{architecture}/{cfg['protocol']['fold']}/seed42",
    })


def serialize_scientific_configuration(cfg):
    return json.dumps(scientific_configuration(cfg), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def configuration_identity(cfg):
    return hashlib.sha256(serialize_scientific_configuration(cfg)).hexdigest()


def operational_configuration(cfg):
    """Separate execution information; callers may persist it in later manifests."""
    _contract(cfg)
    return copy.deepcopy({"experiment": cfg["experiment"], "system": cfg["system"],
                          "logging": cfg.get("logging", {}), "data_root": cfg["data"]["root"],
                          "source_roots": [s.get("root", cfg["data"]["root"]) for s in cfg["data"]["train"]["dataset"]["sources"]],
                          "target_root": cfg["data"]["test"]["dataset"].get("root", cfg["data"]["root"]),
                          "resume": cfg["train"]["save"].get("resume", ""),
                          "weights_path": cfg["model"].get("weights_path"),
                          "num_workers": cfg["data"].get("num_workers"),
                          "pin_memory": cfg["data"].get("pin_memory")})
