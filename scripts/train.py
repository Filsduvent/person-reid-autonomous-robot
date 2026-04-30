import argparse
import json
import logging
import os
import os.path as osp
import sys
from pathlib import Path

import torch

from reid.data.build import build_test_loader, build_train_loader
from reid.engine.train_loop import train_one_epoch
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim.build import build_center_optimizer, build_optimizer, build_scheduler
from reid.utils.checkpoint import load_checkpoint, save_checkpoint
from reid.utils.config import load_config, save_yaml, validate_reid_config
from reid.utils.device import device_summary, select_device
from reid.utils.io import ensure_dir
from reid.utils.logger import setup_logger
from reid.utils.seed import set_seed


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
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Optional config overrides as key=value pairs.",
    )
    return parser.parse_args()


def save_resolved_config(exp_dir, cfg):
    resolved_path = osp.join(exp_dir, "config.resolved.yaml")
    save_yaml(cfg, resolved_path)
    # Keep the shorter alias for compatibility with existing tooling.
    save_yaml(cfg, osp.join(exp_dir, "config.yaml"))
    return resolved_path


def get_checkpoint_paths(checkpoints_dir):
    return {
        "best": osp.join(checkpoints_dir, "ckpt_best.pth"),
        "last": osp.join(checkpoints_dir, "ckpt_last.pth"),
    }


def setup_log_streams(logs_dir):
    ensure_dir(logs_dir)
    stdout_path = osp.join(logs_dir, "stdout.txt")
    stderr_path = osp.join(logs_dir, "stderr.txt")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_stream = TeeStream(original_stdout, stdout_path)
    stderr_stream = TeeStream(original_stderr, stderr_path)
    sys.stdout = stdout_stream
    sys.stderr = stderr_stream
    return original_stdout, original_stderr, stdout_stream, stderr_stream


def log_eval_metrics(tb_writer, scores, epoch):
    if tb_writer is None:
        return

    tb_writer.add_scalar("eval/mAP", float(scores["mAP"]), global_step=epoch)
    tb_writer.add_scalar("eval/mINP", float(scores["mINP"]), global_step=epoch)
    if scores.get("Rank1") is not None:
        tb_writer.add_scalar("eval/Rank1", float(scores["Rank1"]), global_step=epoch)
    if scores.get("Rank5") is not None:
        tb_writer.add_scalar("eval/Rank5", float(scores["Rank5"]), global_step=epoch)
    if scores.get("Rank10") is not None:
        tb_writer.add_scalar("eval/Rank10", float(scores["Rank10"]), global_step=epoch)

    if "rerank_mAP" in scores:
        tb_writer.add_scalar("eval/rerank_mAP", float(scores["rerank_mAP"]), global_step=epoch)
    if "rerank_mINP" in scores:
        tb_writer.add_scalar("eval/rerank_mINP", float(scores["rerank_mINP"]), global_step=epoch)
    if scores.get("rerank_Rank1") is not None:
        tb_writer.add_scalar("eval/rerank_Rank1", float(scores["rerank_Rank1"]), global_step=epoch)
    if scores.get("rerank_Rank5") is not None:
        tb_writer.add_scalar("eval/rerank_Rank5", float(scores["rerank_Rank5"]), global_step=epoch)
    if scores.get("rerank_Rank10") is not None:
        tb_writer.add_scalar("eval/rerank_Rank10", float(scores["rerank_Rank10"]), global_step=epoch)


def log_startup(logger, args, cfg, device, train_loader, num_classes, optimizer, scheduler, resume_path):
    logger.info("Config path: %s", args.config)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Device: %s", device_summary(device))
    logger.info("Train dataset: %s", cfg["data"]["train"]["dataset"]["name"])
    logger.info("Test dataset: %s", cfg["data"]["test"]["dataset"]["name"])
    logger.info("Num classes: %s", num_classes)
    logger.info("Train steps per epoch: %d", len(train_loader))
    logger.info("Optimizer: %s", optimizer.__class__.__name__)
    logger.info("Scheduler: %s", scheduler.__class__.__name__ if scheduler is not None else "None")
    if resume_path:
        logger.info("Resume checkpoint: %s", resume_path)
    logger.info("Resolved config:\n%s", json.dumps(cfg, indent=2))


def create_experiment_dirs(exp_dir):
    ensure_dir(exp_dir)
    metrics_dir = osp.join(exp_dir, "metrics")
    checkpoints_dir = osp.join(exp_dir, "checkpoints")
    logs_dir = osp.join(exp_dir, "logs")
    plots_dir = osp.join(exp_dir, "plots")
    ensure_dir(metrics_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)
    return metrics_dir, checkpoints_dir, logs_dir, plots_dir


def setup_tensorboard(cfg, exp_dir):
    tb = None
    if cfg["logging"]["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = osp.join(exp_dir, "tb")
        ensure_dir(tb_dir)
        tb = SummaryWriter(tb_dir)
    return tb


def maybe_resume_training(logger, cfg, model, optimizer, scheduler, center_optimizer, device):
    resume_path = str(cfg["train"]["save"].get("resume", "") or "").strip()
    if not resume_path:
        return 1, -1.0, None

    checkpoint = load_checkpoint(
        path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        center_optimizer=center_optimizer,
        map_location=device,
    )
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    scores = checkpoint.get("scores")
    best_metric = -1.0
    metric_name = cfg["train"]["save"]["metric"]
    if isinstance(scores, dict) and metric_name in scores:
        best_metric = float(scores[metric_name])
    logger.info("Resumed from epoch %d", int(checkpoint.get("epoch", 0)))
    return start_epoch, best_metric, checkpoint


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


def save_eval_metrics(metrics_dir, cfg, epoch, scores, checkpoint_name=""):
    latest_path = osp.join(metrics_dir, "latest_val.json")
    epoch_path = osp.join(metrics_dir, f"val_epoch_{epoch:03d}.json")
    payload = build_eval_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(epoch_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_training_plots(plots_dir, topk, loss_history, eval_history, logger):
    if not loss_history and not eval_history:
        return

    mpl_dir = osp.join(plots_dir, ".mplconfig")
    ensure_dir(mpl_dir)
    old_mpl_dir = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = mpl_dir

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping plot generation because matplotlib is unavailable: %s", exc)
        if old_mpl_dir is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl_dir
        return

    try:
        if loss_history:
            fig, ax = plt.subplots()
            ax.plot([item["epoch"] for item in loss_history], [item["avg_loss"] for item in loss_history], marker="o")
            ax.set_title("Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(osp.join(plots_dir, "loss_curve.png"))
            plt.close(fig)

        if eval_history:
            epochs = [item["epoch"] for item in eval_history]

            fig, ax = plt.subplots()
            ax.plot(epochs, [item["scores"]["Rank1"] for item in eval_history], marker="o", label="Rank1")
            if any("rerank_Rank1" in item["scores"] for item in eval_history):
                rerank_rank1 = [
                    item["scores"].get("rerank_Rank1", item["scores"]["Rank1"])
                    for item in eval_history
                ]
                ax.plot(epochs, rerank_rank1, marker="o", label="rerank_Rank1")
            ax.set_title("Rank-1 Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rank-1")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(osp.join(plots_dir, "rank1_curve.png"))
            plt.close(fig)

            latest_scores = eval_history[-1]["scores"]

            fig, ax = plt.subplots()
            labels = ["mAP"]
            values = [latest_scores["mAP"]]
            if "rerank_mAP" in latest_scores:
                labels.append("rerank_mAP")
                values.append(latest_scores["rerank_mAP"])
            ax.bar(labels, values)
            ax.set_title("mAP Comparison")
            ax.set_ylabel("Score")
            fig.tight_layout()
            fig.savefig(osp.join(plots_dir, "map_bar.png"))
            plt.close(fig)

            fig, ax = plt.subplots()
            labels = ["mINP"]
            values = [latest_scores["mINP"]]
            if "rerank_mINP" in latest_scores:
                labels.append("rerank_mINP")
                values.append(latest_scores["rerank_mINP"])
            ax.bar(labels, values)
            ax.set_title("mINP Comparison")
            ax.set_ylabel("Score")
            fig.tight_layout()
            fig.savefig(osp.join(plots_dir, "minp_bar.png"))
            plt.close(fig)

            cmc_scores = latest_scores.get("cmc", [])
            rerank_cmc_scores = latest_scores.get("rerank_cmc", [])
            if cmc_scores:
                fig, ax = plt.subplots()
                ax.plot(topk[:len(cmc_scores)], cmc_scores, marker="o", label="CMC")
                if rerank_cmc_scores:
                    ax.plot(topk[:len(rerank_cmc_scores)], rerank_cmc_scores, marker="o", label="rerank_CMC")
                ax.set_title("CMC Curve")
                ax.set_xlabel("Rank")
                ax.set_ylabel("Accuracy")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(osp.join(plots_dir, "cmc_curve.png"))
                plt.close(fig)
    finally:
        if old_mpl_dir is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl_dir


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    validate_reid_config(cfg)

    from reid.engine.evaluator import evaluate_reid

    exp_dir = resolve_repo_relative_path(cfg["experiment"]["output_dir"])
    cfg["experiment"]["output_dir"] = exp_dir
    metrics_dir, checkpoints_dir, logs_dir, plots_dir = create_experiment_dirs(exp_dir)
    checkpoint_paths = get_checkpoint_paths(checkpoints_dir)
    original_stdout, original_stderr, stdout_stream, stderr_stream = setup_log_streams(logs_dir)

    logger = None
    tb = None
    try:
        logger = setup_logger("reid.train", exp_dir, filename="train.log")
        resolved_config_path = save_resolved_config(exp_dir, cfg)

        device, _ = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0), cfg)
        set_seed(
            seed=int(cfg["repro"]["seed"]),
            deterministic=bool(cfg["repro"]["deterministic"]),
            benchmark=bool(cfg["repro"]["benchmark"]),
        )

        tb = setup_tensorboard(cfg, exp_dir)

        train_loader, batch_size = build_train_loader(cfg)
        train_dataset = train_loader.dataset

        classifier_enabled = bool(cfg["model"]["head"].get("classifier", False))
        center_enabled = bool(cfg["loss"].get("center", {}).get("enabled"))
        num_classes = None
        if classifier_enabled or bool(cfg["loss"].get("id", {}).get("enabled")) or center_enabled:
            num_classes = getattr(train_dataset, "num_classes", None)
            if num_classes is None:
                raise ValueError(
                    "Classifier, ID loss, or center loss enabled but train dataset has no "
                    "'num_classes' attribute."
                )
            validate_reid_config(cfg, num_classes=num_classes)

        test_loader = build_test_loader(cfg)

        model = build_model(cfg, num_classes=num_classes).to(device)
        feat_dim = getattr(model, "feat_dim", None)
        if feat_dim is None or int(feat_dim) <= 0:
            raise ValueError("Model must expose a positive 'feat_dim' attribute for loss construction.")

        criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=feat_dim).to(device)
        optimizer = build_optimizer(cfg, model)
        center_optimizer = build_center_optimizer(cfg, criterion)
        scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

        resume_path = str(cfg["train"]["save"].get("resume", "") or "").strip()
        log_startup(logger, args, cfg, device, train_loader, num_classes, optimizer, scheduler, resume_path)
        logger.info("Train batch size: %s", batch_size)
        logger.info("Resolved config saved to: %s", resolved_config_path)
        logger.info("Raw stdout saved to: %s", osp.join(logs_dir, "stdout.txt"))
        logger.info("Raw stderr saved to: %s", osp.join(logs_dir, "stderr.txt"))

        start_epoch, best_metric, _ = maybe_resume_training(
            logger,
            cfg,
            model,
            optimizer,
            scheduler,
            center_optimizer,
            device,
        )

        amp = bool(cfg["system"]["amp"])
        log_interval = int(cfg["system"]["log_interval"])
        epochs = int(cfg["train"]["epochs"])
        eval_interval = int(cfg["train"]["eval_interval"])
        best_name = cfg["train"]["save"]["metric"]
        save_best = bool(cfg["train"]["save"].get("save_best", True))
        save_last = bool(cfg["train"]["save"].get("save_last", True))
        loss_history = []
        eval_history = []

        for ep in range(start_epoch, epochs + 1):
            avg_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                center_optimizer=center_optimizer,
                device=device,
                amp=amp,
                log_interval=log_interval,
                scheduler=scheduler,
                tb_writer=tb,
                epoch=ep,
                logger=logger,
            )
            loss_history.append({"epoch": ep, "avg_loss": float(avg_loss)})
            logger.info("[Epoch %d] avg_loss=%.4f", ep, avg_loss)

            latest_scores = None
            if (ep % eval_interval) == 0:
                scores = evaluate_reid(cfg, model, test_loader, device, logger=logger)
                latest_scores = scores
                eval_history.append({"epoch": ep, "scores": scores})
                logger.info(
                    "[Eval] epoch=%d mAP=%.4f Rank1=%.4f mINP=%.4f",
                    ep,
                    scores["mAP"],
                    scores["Rank1"],
                    scores["mINP"],
                )
                log_eval_metrics(tb, scores, ep)
                save_eval_metrics(metrics_dir, cfg, ep, scores, checkpoint_name=osp.basename(checkpoint_paths["last"]))
                save_training_plots(
                    plots_dir=plots_dir,
                    topk=list(cfg["eval"]["topk"]),
                    loss_history=loss_history,
                    eval_history=eval_history,
                    logger=logger,
                )

                if save_best:
                    cur = float(scores[best_name])
                    if cur > best_metric:
                        best_metric = cur
                        best_path = checkpoint_paths["best"]
                        save_checkpoint(
                            path=best_path,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            center_optimizer=center_optimizer,
                            epoch=ep,
                            scores=scores,
                            cfg=cfg,
                            is_best=True,
                        )
                        logger.info("[CKPT] New best %s=%.4f saved: %s", best_name, best_metric, best_path)

            if save_last:
                last_path = checkpoint_paths["last"]
                save_checkpoint(
                    path=last_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    center_optimizer=center_optimizer,
                    epoch=ep,
                    scores=latest_scores,
                    cfg=cfg,
                )

    finally:
        if tb is not None:
            tb.close()
        logging.shutdown()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_stream.close()
        stderr_stream.close()


if __name__ == "__main__":
    main()
