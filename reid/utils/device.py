# reid/utils/device.py
from __future__ import annotations

import torch
from typing import Any, Dict, Tuple


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        return f"cuda:{idx} | {name} | capability={cap}"
    return "cpu"


def _force_cpu_policy(cfg: Dict[str, Any]) -> None:
    # CPU policy: no AMP, no pin_memory
    cfg.setdefault("system", {})
    cfg["system"]["amp"] = False

    # Option-3 schema: data.pin_memory (global)
    data = cfg.get("data", {})
    if isinstance(data, dict):
        data["pin_memory"] = False
        cfg["data"] = data


def select_device(device_str: str, gpu_id: int, cfg: Dict[str, Any] | None = None) -> Tuple[torch.device, Dict[str, Any] | None]:
    """
    device_str: auto|cpu|cuda
    If cfg is provided, apply device policy in-place (safe modifications).
    """
    device_str = (device_str or "auto").lower()

    if device_str == "auto":
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            # hard fallback to cpu
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu_id}")
    elif device_str == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported system.device='{device_str}' (use auto|cpu|cuda)")

    if cfg is not None and device.type == "cpu":
        _force_cpu_policy(cfg)

    return device, cfg
