from typing import Tuple
import torch


def select_device(device_policy: str, gpu_id: int = 0) -> torch.device:
    """
    device_policy: 'auto' | 'cpu' | 'cuda'
    """
    device_policy = device_policy.lower()
    if device_policy not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Invalid device policy: {device_policy}")

    if device_policy == "cpu":
        return torch.device("cpu")

    if device_policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{gpu_id}")

    # auto
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        return f"CUDA:{idx} ({name})"
    return "CPU"
