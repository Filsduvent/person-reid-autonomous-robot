# reid/utils/config.py
from __future__ import annotations

import os
import re
import copy
import yaml
from typing import Any, Dict, List, Tuple


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


def save_yaml(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
