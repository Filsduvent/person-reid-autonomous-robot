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


def validate_reid_config(cfg: Dict[str, Any], num_classes: int | None = None) -> None:
    model_cfg = cfg.get("model", {})
    head_cfg = model_cfg.get("head", {})
    loss_cfg = cfg.get("loss", {})

    classifier_enabled = bool(head_cfg.get("classifier", False))
    metric_feat = _expect_choice("model.head.metric_feat", head_cfg.get("metric_feat", "bn"), {"raw", "bn"})

    id_cfg = loss_cfg.get("id", {})
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
