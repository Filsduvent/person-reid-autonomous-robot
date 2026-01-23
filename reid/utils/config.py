import os
import re
from copy import deepcopy
from typing import Any, Dict

import yaml


_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Config variable path not found: {path}")
        cur = cur[part]
    return cur


def _resolve_vars_in_str(s: str, cfg: Dict[str, Any]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        val = _get_by_path(cfg, key)
        return str(val)
    return _VAR_PATTERN.sub(repl, s)


def _walk_and_resolve(obj: Any, cfg: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v, cfg) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(v, cfg) for v in obj]
    if isinstance(obj, str):
        return _resolve_vars_in_str(obj, cfg)
    return obj


def load_config(path: str) -> Dict[str, Any]:
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve variables using a two-pass strategy
    cfg = deepcopy(cfg)
    cfg = _walk_and_resolve(cfg, cfg)

    # expand user in paths
    if "experiment" in cfg and "output_dir" in cfg["experiment"]:
        cfg["experiment"]["output_dir"] = os.path.expanduser(cfg["experiment"]["output_dir"])
    if "dataset" in cfg and "root" in cfg["dataset"]:
        cfg["dataset"]["root"] = os.path.expanduser(cfg["dataset"]["root"])

    return cfg
