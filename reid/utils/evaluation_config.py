"""Source-model, target-dataset and execution ownership for evaluation only."""
import copy


def build_evaluation_config(source_cfg, data_cfg, output_dir):
    """Copy source semantics, replacing only dataset and supported execution fields.

    The complete test.dataset mapping owns parser/protocol options. All other
    test fields stay source-owned except batch.size. In particular, images,
    aug, loader ordering and the eval policy are never imported from a target.
    No training configuration is consumed to construct a model or a loader here.
    """
    cfg = copy.deepcopy(source_cfg)
    cfg["data"]["root"] = copy.deepcopy(data_cfg["data"]["root"])
    cfg["data"]["test"]["dataset"] = copy.deepcopy(data_cfg["data"]["test"]["dataset"])
    for key in ("num_workers", "pin_memory"):
        if key in data_cfg["data"]:
            cfg["data"][key] = copy.deepcopy(data_cfg["data"][key])
    batch = data_cfg["data"]["test"].get("batch", {})
    if "size" in batch:
        cfg["data"]["test"].setdefault("batch", {})["size"] = copy.deepcopy(batch["size"])
    for key in ("device", "gpu_id"):
        if key in data_cfg.get("system", {}):
            cfg.setdefault("system", {})[key] = copy.deepcopy(data_cfg["system"][key])
    cfg["experiment"]["output_dir"] = str(output_dir)
    return cfg


def prepare_standalone_config(checkpoint, requested_cfg):
    """Use checkpoint semantics; reject conflicting standalone model/input policy.

    Old/raw checkpoints without cfg retain the caller-supplied source-config
    fallback. Reconstruction still validates metadata and strictly loads weights.
    Comparisons are intentionally conservative: supply the saved source config
    plus dataset/runtime overrides, or use the cross-domain entry point.
    """
    source = checkpoint.get("cfg")
    if source is None:
        return copy.deepcopy(requested_cfg)
    if not isinstance(source, dict):
        raise ValueError("Checkpoint cfg must be a resolved source configuration mapping.")
    sections = {
        "model": (source["model"], requested_cfg["model"]),
        "data.test.images": (source["data"]["test"]["images"], requested_cfg["data"]["test"]["images"]),
        "data.test.aug": (source["data"]["test"]["aug"], requested_cfg["data"]["test"]["aug"]),
        "data.test.loader": (source["data"]["test"]["loader"], requested_cfg["data"]["test"]["loader"]),
        "eval": tuple({k: v for k, v in c["eval"].items() if k not in {"weight", "export"}}
                      for c in (source, requested_cfg)),
    }
    for name, (saved, supplied) in sections.items():
        if saved != supplied:
            raise ValueError(f"Standalone evaluation config conflicts with checkpoint source {name}. "
                             "Use the saved source configuration with dataset/runtime overrides; "
                             "use evaluate_cross_domain.py for a target preset.")
    return build_evaluation_config(source, requested_cfg, requested_cfg["experiment"]["output_dir"])
