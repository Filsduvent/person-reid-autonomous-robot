from __future__ import annotations

import json
import os
from pathlib import Path

from reid.utils.io import ensure_dir


CORE_METRIC_KEYS = ("mAP", "mINP", "Rank1", "Rank5", "Rank10")
RERANK_METRIC_KEYS = ("rerank_mAP", "rerank_mINP", "rerank_Rank1", "rerank_Rank5", "rerank_Rank10")


def _model_label(cfg):
    model_cfg = cfg.get("model", {})
    name = str(model_cfg.get("name", "unknown"))
    backbone_name = str(model_cfg.get("backbone", {}).get("name", ""))
    if name == "reid_baseline" and backbone_name:
        return f"{backbone_name}_baseline"
    return name


def build_metric_payload(cfg, scores, epoch=None, checkpoint_name=""):
    payload = {
        "dataset": cfg["data"]["test"]["dataset"]["name"],
        "split": cfg["data"]["test"]["dataset"]["split"],
        "model": _model_label(cfg),
        "epoch": epoch,
        "checkpoint": checkpoint_name,
    }

    for key in CORE_METRIC_KEYS:
        payload[key] = float(scores[key]) if scores.get(key) is not None else None

    for key in RERANK_METRIC_KEYS:
        if key in scores:
            payload[key] = float(scores[key]) if scores.get(key) is not None else None

    return payload


def write_metric_payload(path, payload):
    path = Path(path)
    ensure_dir(str(path.parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def save_train_eval_metrics(metrics_dir, cfg, epoch, scores, checkpoint_name=""):
    ensure_dir(metrics_dir)
    payload = build_metric_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)
    latest_path = os.path.join(metrics_dir, "latest_val.json")
    epoch_path = os.path.join(metrics_dir, f"val_epoch_{epoch:03d}.json")
    write_metric_payload(latest_path, payload)
    write_metric_payload(epoch_path, payload)
    return {
        "latest": latest_path,
        "epoch": epoch_path,
        "payload": payload,
    }


def save_final_eval_metrics(metrics_dir, cfg, epoch, scores, checkpoint_name=""):
    ensure_dir(metrics_dir)
    payload = build_metric_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)
    final_path = os.path.join(metrics_dir, "final_test.json")
    write_metric_payload(final_path, payload)
    return {
        "final": final_path,
        "payload": payload,
    }


def save_final_epoch_metrics(metrics_dir, cfg, epoch, scores, checkpoint_name="ckpt_last.pth"):
    """Persist metrics measured on the final training epoch without ambiguity."""
    ensure_dir(metrics_dir)
    payload = build_metric_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)
    path = os.path.join(metrics_dir, "final_epoch_test.json")
    write_metric_payload(path, payload)
    return {"final_epoch": path, "payload": payload}


def save_standalone_eval_metrics(metrics_dir, cfg, checkpoint_name, scores, epoch=None):
    ensure_dir(metrics_dir)
    payload = build_metric_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)
    stem = Path(checkpoint_name).stem
    eval_path = os.path.join(metrics_dir, f"eval_{stem}.json")
    final_path = os.path.join(metrics_dir, "final_test.json")
    write_metric_payload(eval_path, payload)
    write_metric_payload(final_path, payload)
    return {
        "eval": eval_path,
        "final": final_path,
        "payload": payload,
    }
