import copy
import os.path as osp
from collections import OrderedDict
from collections.abc import Mapping

import torch

from reid.utils.io import ensure_dir


RECONSTRUCTION_VERSION = 1
OUTPUT_CONTRACT_VERSION = 1


def _positive_int(value, name, optional=False):
    if optional and value is None:
        return
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer" + (" or None" if optional else ""))


def make_reconstruction_metadata(cfg, num_classes, embedding_dim):
    """Model builders declare dimensions explicitly; never inspect arbitrary heads.

    Version 1 describes the dictionary-output model contract. cfg remains the
    single canonical configuration in the checkpoint, not a metadata copy.
    """
    metadata = {
        "schema_version": RECONSTRUCTION_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "model_name": cfg["model"]["name"],
        "variant": cfg["model"].get("variant"),
        "num_classes": num_classes,
        "embedding_dim": embedding_dim,
    }
    _validate_metadata(metadata, cfg)
    return metadata


def _validate_metadata(metadata, cfg):
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint reconstruction metadata must be a mapping.")
    for key, expected in (("schema_version", RECONSTRUCTION_VERSION),
                          ("output_contract_version", OUTPUT_CONTRACT_VERSION)):
        if type(metadata.get(key)) is not int or metadata[key] != expected:
            raise ValueError(f"Unsupported reconstruction {key}: {metadata.get(key)!r}")
    required = {"model_name", "variant", "num_classes", "embedding_dim"}
    if not required.issubset(metadata):
        raise ValueError(f"Missing reconstruction fields: {sorted(required - metadata.keys())}")
    if not isinstance(cfg, Mapping) or not isinstance(cfg.get("model"), Mapping):
        raise ValueError("Reconstruction metadata requires canonical checkpoint cfg.model.")
    mcfg = cfg["model"]
    if not isinstance(metadata["model_name"], str) or not metadata["model_name"]:
        raise ValueError("reconstruction model_name must be a nonempty string")
    if metadata["variant"] is not None and not isinstance(metadata["variant"], str):
        raise ValueError("reconstruction variant must be a string or None")
    for key, config_value in (("model_name", mcfg.get("name")),
                              ("variant", mcfg.get("variant"))):
        if metadata[key] != config_value:
            raise ValueError(f"Reconstruction metadata/config mismatch: {key}")
    _positive_int(metadata["num_classes"], "num_classes", optional=True)
    _positive_int(metadata["embedding_dim"], "embedding_dim")
    # Optional source-model dimensions in cfg must agree. Dataset/target class
    # counts are deliberately never consulted.
    for section in (mcfg, mcfg.get("head", {})):
        if not isinstance(section, Mapping):
            raise ValueError("Reconstruction cfg.model/head must be mappings.")
        for key in ("num_classes", "embedding_dim"):
            if key in section and section[key] != metadata[key]:
                raise ValueError(f"Reconstruction metadata/config mismatch: {key}")


def normalized_model_state(checkpoint):
    """Accept wrapped/raw states; strip only one uniform DataParallel prefix."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping.")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(k, str) for k in state):
        raise ValueError("Checkpoint model state must be a string-keyed mapping.")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed module. prefixes in checkpoint model state.")
    if prefixed and all(prefixed):
        normalized = OrderedDict((key[7:], value) for key, value in state.items())
        if hasattr(state, "_metadata"):
            normalized._metadata = {
                ("" if key == "module" else key[7:]): value
                for key, value in state._metadata.items()
                if key == "module" or key.startswith("module.")
            }
        return normalized
    return state


def infer_num_classes_from_checkpoint(checkpoint):
    """Metadata first; classifier.weight is a bounded historical baseline fallback."""
    state = normalized_model_state(checkpoint)
    if "reconstruction" in checkpoint:
        metadata = checkpoint["reconstruction"]
        _validate_metadata(metadata, checkpoint.get("cfg"))
        return metadata["num_classes"]
    cfg = checkpoint.get("cfg")
    if cfg is not None and (not isinstance(cfg, Mapping) or
                            cfg.get("model", {}).get("name") != "reid_baseline"):
        raise ValueError("Legacy reconstruction is supported only for reid_baseline.")
    weight = state.get("classifier.weight")
    if weight is None:
        return None
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise ValueError("Legacy classifier.weight must be a two-dimensional tensor.")
    return int(weight.shape[0])


def reconstruct_model(checkpoint, cfg=None):
    """Build and strictly load a source model, without pretrained initialization.

    Embedded cfg is authoritative. cfg is used only for historical raw states
    or old checkpoints without configuration; it is never a source of classes.
    Evaluation data/preprocessing remain the caller's responsibility.
    """
    from reid.models.build import build_model

    state = normalized_model_state(checkpoint)
    num_classes = infer_num_classes_from_checkpoint(checkpoint)
    source_cfg = checkpoint.get("cfg")
    source_cfg = copy.deepcopy(source_cfg if source_cfg is not None else cfg)
    if not isinstance(source_cfg, dict) or not isinstance(source_cfg.get("model"), dict):
        raise ValueError("Model reconstruction requires a source model configuration.")
    metadata = checkpoint.get("reconstruction")
    if metadata is None and source_cfg["model"].get("name") != "reid_baseline":
        raise ValueError("Legacy reconstruction is supported only for reid_baseline.")
    model = build_model(source_cfg, num_classes=num_classes, initialize_pretrained=False)
    if metadata is not None:
        declared = getattr(model, "checkpoint_metadata", None)
        if declared != metadata:
            raise ValueError("Reconstruction metadata does not match the constructed model declaration.")
    model.load_state_dict(state, strict=True)
    return model


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

    wrappers = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    source_model = model.module if isinstance(model, wrappers) else model
    metadata = getattr(source_model, "checkpoint_metadata", None)
    # Keep the historical optional-cfg save API: without cfg this remains a
    # weights/training-state checkpoint, not a self-describing reconstruction.
    if metadata is not None and cfg is not None:
        _validate_metadata(metadata, cfg)
        payload["reconstruction"] = copy.deepcopy(metadata)

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
    model_state = normalized_model_state(checkpoint)
    if "reconstruction" in checkpoint:
        _validate_metadata(checkpoint["reconstruction"], checkpoint.get("cfg"))
    model.load_state_dict(model_state, strict=True)

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
