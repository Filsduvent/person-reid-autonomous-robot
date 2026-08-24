"""Evaluate a source-domain checkpoint directly on a target ReID dataset."""

from __future__ import annotations

import argparse
import copy
import os.path as osp
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reid.data.build import build_test_loader
from reid.engine.evaluator import evaluate_reid
from reid.models.build import build_model
from reid.utils.checkpoint import load_checkpoint
from reid.utils.config import load_config, save_yaml, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.device import select_device
from reid.utils.experiment_matrix import build_cross_dataset_record, write_cross_dataset_record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Target-dataset experiment YAML.")
    parser.add_argument("--checkpoint", "--weight", dest="checkpoint", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def infer_num_classes(checkpoint):
    weight = checkpoint.get("model", {}).get("classifier.weight")
    return int(weight.shape[0]) if weight is not None and weight.ndim == 2 else None


def build_cross_domain_config(source_cfg, target_cfg, target_dataset, output_dir):
    """Keep source architecture/loss settings and substitute target evaluation data only."""
    cfg = copy.deepcopy(source_cfg)
    cfg["data"]["root"] = target_cfg["data"]["root"]
    cfg["data"]["num_workers"] = target_cfg["data"]["num_workers"]
    cfg["data"]["pin_memory"] = target_cfg["data"]["pin_memory"]
    cfg["data"]["test"] = copy.deepcopy(target_cfg["data"]["test"])
    actual_target = cfg["data"]["test"]["dataset"]["name"]
    if actual_target != target_dataset:
        raise ValueError(f"Target config dataset is '{actual_target}', expected '{target_dataset}'.")
    cfg["experiment"]["output_dir"] = str(output_dir)
    return cfg


def main():
    args = parse_args()
    target_cfg = load_config(args.config, overrides=args.opts)
    validate_config(target_cfg)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_cfg = checkpoint.get("cfg")
    if not isinstance(source_cfg, dict):
        raise ValueError("Checkpoint does not contain its resolved source configuration.")
    checkpoint_source = source_cfg["data"]["train"]["dataset"]["name"]
    if checkpoint_source != args.source_dataset:
        raise ValueError(f"Checkpoint source dataset is '{checkpoint_source}', not '{args.source_dataset}'.")

    cfg = build_cross_domain_config(source_cfg, target_cfg, args.target_dataset, args.output_dir)
    validate_config(cfg)
    validate_reid_config(cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output_dir / "config.resolved.yaml")

    device, _ = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0), cfg)
    model = build_model(cfg, num_classes=infer_num_classes(checkpoint)).to(device)
    load_checkpoint(str(checkpoint_path), model=model, map_location=device)
    scores = evaluate_reid(cfg, model, build_test_loader(cfg), device)
    architecture = str(cfg["model"]["name"])
    record = build_cross_dataset_record(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        architecture=architecture,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        scores=scores,
    )
    path = write_cross_dataset_record(output_dir / "cross_dataset.json", record)
    print(f"Saved cross-domain result: {path}")


if __name__ == "__main__":
    main()
