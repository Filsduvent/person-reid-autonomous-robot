# reid/utils/config.py
from __future__ import annotations

import os
import re
import copy
import yaml
from typing import Any, Dict, List, Tuple


def _expect_choice(name: str, value: Any, allowed: set[str]) -> str:
    value = str(value).lower()
    if value not in allowed:
        opts = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {name}='{value}'. Use one of: {opts}.")
    return value


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def _parse_scalar(s: str) -> Any:
    """
    Parse override RHS. Supports:
    - null/None, true/false
    - ints/floats
    - lists/tuples/dicts via YAML
    - strings (default)
    """
    # Use YAML parser for robust types: "0.1", "[1,2]", "{a:1}", "true", "null"
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    overrides format: ["a.b.c=123", "data.train.batch.P=16"]
    """
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Invalid override '{ov}'. Expected key=value.")
        key, val = ov.split("=", 1)
        key = key.strip()
        val = _parse_scalar(val.strip())

        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return cfg


def _expand_user_in_cfg(cfg: Any) -> Any:
    """Recursively expand ~ in string paths."""
    if isinstance(cfg, dict):
        return {k: _expand_user_in_cfg(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [_expand_user_in_cfg(x) for x in cfg]
    if isinstance(cfg, str):
        return os.path.expanduser(cfg)
    return cfg


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _get_by_path(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"Interpolation path not found: {path}")
        cur = cur[p]
    return cur


def resolve_interpolations(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal interpolation resolver for strings like:
      exp/${experiment.name}
    or in our schema:
      exp/${experiment.name}  (still supported)
    """
    cfg = copy.deepcopy(cfg)

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(x) for x in obj]
        if isinstance(obj, str):
            def repl(m):
                path = m.group(1).strip()
                return str(_get_by_path(cfg, path))
            return _VAR_PATTERN.sub(repl, obj)
        return obj

    return _resolve(cfg)


def load_config(path: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    cfg = _expand_user_in_cfg(cfg)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    # resolve ${...} after overrides so experiment.name affects output_dir, etc.
    cfg = resolve_interpolations(cfg)

    return cfg


PCB_VARIANT = "independent_part_reduction"


def validate_pcb_model_config(model_cfg: dict) -> None:
    """Validate the single PCB recipe and persist its default variant in cfg.

    This canonicalization keeps the saved cfg and reconstruction metadata in
    agreement, including when the user selects PCB with model.name alone.
    Architecture internals are fixed in PCB, not a configurable ablation surface.
    """
    allowed = {"name", "variant", "pretrained", "weights_path"}
    unknown = set(model_cfg) - allowed
    if unknown:
        raise ValueError(f"Unsupported PCB model fields: {sorted(unknown)}; "
                         "PCB uses fixed ResNet50/stride1/dilation1/six independent 256-D parts.")
    if model_cfg.get("variant", PCB_VARIANT) != PCB_VARIANT:
        raise ValueError(f"PCB model.variant must be '{PCB_VARIANT}'.")
    if type(model_cfg.get("pretrained", True)) is not bool:
        raise ValueError("PCB model.pretrained must be a boolean.")
    path = model_cfg.get("weights_path")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ValueError("PCB model.weights_path must be a nonempty path string or None.")
    if path is not None and not model_cfg.get("pretrained", True):
        raise ValueError("PCB model.weights_path requires model.pretrained=true.")
    model_cfg.setdefault("variant", PCB_VARIANT)


def model_requires_num_classes(cfg: dict) -> bool:
    """Architecture construction capability, independent of dataset identity."""
    model_cfg = cfg.get("model", {})
    return (model_cfg.get("name") == "pcb"
            or bool(model_cfg.get("head", {}).get("classifier", False)))


def validate_model_loss_requirements(cfg: dict, model):
    """Resolve optional metric width; Center alone needs it at construction.

    Triplet consumes a runtime metric tensor, whose presence is checked by
    LossBundle. Never substitute embedding_dim or silently disable a loss.
    """
    feat_dim = getattr(model, "feat_dim", None)
    if cfg.get("loss", {}).get("center", {}).get("enabled", False):
        if type(feat_dim) is not int or feat_dim <= 0:
            raise ValueError("Center loss requires a positive model.feat_dim metric-feature width.")
    return feat_dim


def validate_reid_config(cfg: Dict[str, Any], num_classes: int | None = None) -> None:
    model_cfg = cfg.get("model", {})
    head_cfg = model_cfg.get("head", {})
    loss_cfg = cfg.get("loss", {})

    classifier_enabled = model_requires_num_classes(cfg)
    if model_cfg.get("name") == "pcb":
        validate_pcb_model_config(model_cfg)
        if any(loss_cfg.get(name, {}).get("enabled", False) for name in ("triplet", "center")):
            raise ValueError("PCB provides no feat_raw/feat_bn metric features; disable Triplet and Center losses.")
        metric_feat = None
    else:
        metric_feat = _expect_choice("model.head.metric_feat", head_cfg.get("metric_feat", "bn"), {"raw", "bn"})

    id_cfg = loss_cfg.get("id", {})
    aggregation = id_cfg.get("head_aggregation", "sum")
    if aggregation not in ("sum", "mean"):
        raise ValueError("loss.id.head_aggregation must be 'sum' or 'mean'.")
    id_enabled = bool(id_cfg.get("enabled", False))
    center_cfg = loss_cfg.get("center", {})
    center_enabled = bool(center_cfg.get("enabled", False))

    if id_enabled and not classifier_enabled:
        raise ValueError("ID loss enabled but model.head.classifier is false.")

    if center_enabled:
        _expect_choice("loss.center.feat", center_cfg.get("feat", "raw"), {"raw", "bn"})

    # Explicitly validate even when not currently used elsewhere so invalid values fail early.
    _ = metric_feat

    if num_classes is not None:
        if (classifier_enabled or id_enabled or center_enabled) and int(num_classes) <= 1:
            raise ValueError("Classifier, ID loss, or center loss enabled but num_classes <= 1.")


def save_yaml(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
