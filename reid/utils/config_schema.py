from __future__ import annotations

from typing import Any


REQUIRED_TOP_LEVEL_SECTIONS = (
    "experiment",
    "system",
    "repro",
    "logging",
    "data",
    "model",
    "loss",
    "optim",
    "sched",
    "train",
    "eval",
)

REQUIRED_KEYS = (
    "experiment.name",
    "experiment.output_dir",
    "data.train.dataset.name",
    "data.test.dataset.name",
    "model.name",
    "optim.name",
    "train.epochs",
    "eval.topk",
)

VALID_SYSTEM_DEVICES = {"auto", "cpu", "cuda"}


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _require_mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{path}' must be a mapping.")
    return value


def _get_required(cfg: dict, dotted_key: str) -> Any:
    current: Any = cfg
    parts = dotted_key.split(".")
    for idx, part in enumerate(parts):
        path = ".".join(parts[: idx + 1])
        if not isinstance(current, dict):
            parent = ".".join(parts[:idx])
            raise ValueError(f"Config section '{parent}' must be a mapping.")
        if part not in current:
            raise ValueError(f"Missing config key: {path}")
        current = current[part]
    if not _is_present(current):
        raise ValueError(f"Config key '{dotted_key}' must not be empty.")
    return current


def validate_config(cfg: dict) -> None:
    """
    Validate the experiment-level YAML schema used by train/evaluate/smoke.

    This intentionally checks structure and required experiment knobs only.
    Model/loss compatibility is handled by validate_reid_config after overrides
    and, when needed, after num_classes is known.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping.")

    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")
        _require_mapping(cfg[section], section)

    for key in REQUIRED_KEYS:
        _get_required(cfg, key)

    system_device = str(_get_required(cfg, "system.device")).lower()
    if system_device not in VALID_SYSTEM_DEVICES:
        allowed = ", ".join(sorted(VALID_SYSTEM_DEVICES))
        raise ValueError(
            f"Invalid config value for system.device: '{system_device}'. "
            f"Expected one of: {allowed}."
        )

    loss_cfg = cfg["loss"]
    if not isinstance(loss_cfg, dict):
        raise ValueError("Config section 'loss' must be a mapping.")
