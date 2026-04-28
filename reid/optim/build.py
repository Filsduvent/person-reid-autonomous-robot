"""Optimizer and scheduler builders for ReID training."""

import torch
from torch.optim.lr_scheduler import MultiStepLR

from reid.optim.lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model):
    """Build an optimizer with Bag of Tricks-style parameter groups."""

    ocfg = cfg["optim"]
    name = ocfg["name"].lower()
    base_lr = float(ocfg["lr"])
    weight_decay = float(ocfg["weight_decay"])
    bias_lr_factor = float(ocfg.get("bias_lr_factor", 1.0))
    weight_decay_bias = float(ocfg.get("weight_decay_bias", weight_decay))
    momentum = float(ocfg.get("momentum", 0.9))
    nesterov = bool(ocfg.get("nesterov", False))

    params = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = base_lr
        param_weight_decay = weight_decay
        if "bias" in param_name:
            lr = base_lr * bias_lr_factor
            param_weight_decay = weight_decay_bias

        params.append(
            {
                "params": [param],
                "lr": lr,
                "weight_decay": param_weight_decay,
                "param_name": param_name,
            }
        )

    if name == "sgd":
        return torch.optim.SGD(params, lr=base_lr, momentum=momentum, nesterov=nesterov)
    if name == "adam":
        return torch.optim.Adam(params, lr=base_lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=base_lr)

    raise ValueError(f"Unknown optimizer: {name}")


def build_center_optimizer(cfg, criterion):
    """Build the optional optimizer for center loss parameters."""

    center_cfg = cfg["loss"].get("center", {})
    if not center_cfg.get("enabled", False):
        return None
    if criterion is None:
        return None

    center_loss = getattr(criterion, "center_loss", None)
    if center_loss is None:
        center_loss = getattr(criterion, "center", None)
    if center_loss is None:
        return None

    center_lr = float(center_cfg["lr"])
    params = [param for param in center_loss.parameters() if param.requires_grad]
    if not params:
        return None

    return torch.optim.SGD(params, lr=center_lr)


def build_scheduler(cfg, optimizer):
    """Build the learning-rate scheduler from YAML config."""

    sched_cfg = cfg.get("sched")
    if not sched_cfg:
        return None

    name = str(sched_cfg.get("name", "")).lower()
    if name in {"", "none", "disabled"}:
        return None

    milestones = sched_cfg.get("milestones", [])
    gamma = float(sched_cfg.get("gamma", 0.1))

    if name == "warmup_multistep":
        return WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
            warmup_factor=float(sched_cfg.get("warmup_factor", 1.0 / 3.0)),
            warmup_iters=int(sched_cfg.get("warmup_iters", 500)),
            warmup_method=str(sched_cfg.get("warmup_method", "linear")).lower(),
        )

    if name == "step":
        return MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    raise ValueError(f"Unknown scheduler: {name}")
