import os.path as osp

import torch

from reid.utils.io import ensure_dir


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    center_optimizer=None,
    epoch=0,
    scores=None,
    cfg=None,
    is_best=False,
):
    del is_best  # path selection remains the caller's responsibility.

    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "center_optimizer": center_optimizer.state_dict() if center_optimizer is not None else None,
        "scores": scores,
        "cfg": cfg,
    }

    path = osp.abspath(path)
    ensure_dir(osp.dirname(path))
    torch.save(payload, path)
    return payload


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    center_optimizer=None,
    map_location="cpu",
):
    checkpoint = torch.load(path, map_location=map_location)
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint

    if model_state and all(str(key).startswith("module.") for key in model_state.keys()):
        model_state = {str(key)[7:]: value for key, value in model_state.items()}

    model.load_state_dict(model_state)

    optimizer_state = checkpoint.get("optimizer")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    center_optimizer_state = checkpoint.get("center_optimizer")
    if center_optimizer is not None and center_optimizer_state is not None:
        center_optimizer.load_state_dict(center_optimizer_state)

    return checkpoint
