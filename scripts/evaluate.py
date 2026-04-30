import argparse
import json
import logging
import os.path as osp
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TeeStream:
    def __init__(self, original, path):
        self.original = original
        self.file = open(path, "a", buffering=1)
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()

    def close(self):
        self.file.close()


def resolve_repo_relative_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Optional config overrides as key=value pairs.",
    )
    return parser.parse_args()


def setup_log_streams(logs_dir):
    from reid.utils.io import ensure_dir

    ensure_dir(logs_dir)
    stdout_path = osp.join(logs_dir, "eval_stdout.txt")
    stderr_path = osp.join(logs_dir, "eval_stderr.txt")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_stream = TeeStream(original_stdout, stdout_path)
    stderr_stream = TeeStream(original_stderr, stderr_path)
    sys.stdout = stdout_stream
    sys.stderr = stderr_stream
    return original_stdout, original_stderr, stdout_stream, stderr_stream


def save_resolved_config(exp_dir, cfg, save_yaml):
    resolved_path = osp.join(exp_dir, "config.resolved.yaml")
    save_yaml(cfg, resolved_path)
    save_yaml(cfg, osp.join(exp_dir, "config.yaml"))
    return resolved_path


def infer_num_classes_from_checkpoint(checkpoint):
    model_state = checkpoint.get("model", {})
    classifier_weight = model_state.get("classifier.weight")
    if classifier_weight is None:
        return None
    if classifier_weight.ndim != 2:
        return None
    return int(classifier_weight.shape[0])


def build_eval_payload(cfg, scores, epoch=None, checkpoint_name=""):
    payload = {
        "dataset": cfg["data"]["test"]["dataset"]["name"],
        "split": cfg["data"]["test"]["dataset"]["split"],
        "epoch": epoch,
        "checkpoint": checkpoint_name,
        "mAP": float(scores["mAP"]),
        "mINP": float(scores["mINP"]),
        "Rank1": float(scores["Rank1"]) if scores.get("Rank1") is not None else None,
        "Rank5": float(scores["Rank5"]) if scores.get("Rank5") is not None else None,
        "Rank10": float(scores["Rank10"]) if scores.get("Rank10") is not None else None,
    }
    if "rerank_mAP" in scores:
        payload.update({
            "rerank_mAP": float(scores["rerank_mAP"]),
            "rerank_mINP": float(scores["rerank_mINP"]),
            "rerank_Rank1": float(scores["rerank_Rank1"]) if scores.get("rerank_Rank1") is not None else None,
            "rerank_Rank5": float(scores["rerank_Rank5"]) if scores.get("rerank_Rank5") is not None else None,
            "rerank_Rank10": float(scores["rerank_Rank10"]) if scores.get("rerank_Rank10") is not None else None,
        })
    return payload


def save_eval_metrics(metrics_dir, cfg, checkpoint_name, scores, epoch=None):
    stem = Path(checkpoint_name).stem
    out_path = osp.join(metrics_dir, f"eval_{stem}.json")
    payload = build_eval_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main():
    args = parse_args()

    from reid.data.build import build_test_loader
    from reid.engine.evaluator import evaluate_reid
    from reid.models.build import build_model
    from reid.utils.checkpoint import load_checkpoint
    from reid.utils.config import load_config, save_yaml, validate_reid_config
    from reid.utils.device import device_summary, select_device
    from reid.utils.io import ensure_dir
    from reid.utils.logger import setup_logger

    cfg = load_config(args.config, overrides=args.opts)
    validate_reid_config(cfg)

    exp_dir = resolve_repo_relative_path(cfg["experiment"]["output_dir"])
    cfg["experiment"]["output_dir"] = exp_dir
    metrics_dir = osp.join(exp_dir, "metrics")
    logs_dir = osp.join(exp_dir, "logs")
    ensure_dir(exp_dir)
    ensure_dir(metrics_dir)
    ensure_dir(logs_dir)
    original_stdout, original_stderr, stdout_stream, stderr_stream = setup_log_streams(logs_dir)

    logger = None
    try:
        logger = setup_logger("reid.evaluate", exp_dir, filename="evaluate.log")
        resolved_config_path = save_resolved_config(exp_dir, cfg, save_yaml)

        device, _ = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0), cfg)
        logger.info("Config path: %s", args.config)
        logger.info("Device: %s", device_summary(device))
        logger.info("Resolved config saved to: %s", resolved_config_path)
        logger.info("Raw eval stdout saved to: %s", osp.join(logs_dir, "eval_stdout.txt"))
        logger.info("Raw eval stderr saved to: %s", osp.join(logs_dir, "eval_stderr.txt"))

        weight_path = args.weight or cfg["eval"].get("weight", "")
        if not weight_path:
            raise ValueError("No checkpoint path provided. Use --weight or set eval.weight in the config.")
        weight_path = resolve_repo_relative_path(weight_path)
        logger.info("Evaluation checkpoint: %s", weight_path)

        checkpoint = torch.load(weight_path, map_location=device)
        num_classes = infer_num_classes_from_checkpoint(checkpoint)

        model = build_model(cfg, num_classes=num_classes).to(device)
        load_checkpoint(weight_path, model=model, map_location=device)

        test_loader = build_test_loader(cfg)
        logger.info("Test dataset: %s", cfg["data"]["test"]["dataset"]["name"])
        logger.info("Resolved config:\n%s", json.dumps(cfg, indent=2))

        scores = evaluate_reid(cfg, model, test_loader, device, logger=logger)
        epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

        logger.info("mAP=%.4f mINP=%.4f Rank1=%.4f Rank5=%.4f Rank10=%.4f",
                    scores["mAP"], scores["mINP"], scores["Rank1"], scores["Rank5"], scores["Rank10"])
        if "rerank_mAP" in scores:
            logger.info(
                "rerank_mAP=%.4f rerank_mINP=%.4f rerank_Rank1=%.4f rerank_Rank5=%.4f rerank_Rank10=%.4f",
                scores["rerank_mAP"],
                scores["rerank_mINP"],
                scores["rerank_Rank1"],
                scores["rerank_Rank5"],
                scores["rerank_Rank10"],
            )

        out_path = save_eval_metrics(metrics_dir, cfg, Path(weight_path).name, scores, epoch=epoch)
        logger.info("Saved metrics to: %s", out_path)
    finally:
        logging.shutdown()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_stream.close()
        stderr_stream.close()


if __name__ == "__main__":
    main()
