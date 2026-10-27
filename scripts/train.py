import argparse
import copy
import json
import logging
import os
import os.path as osp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Capture the launching workspace environment before importing CUDA-dependent code.
from reid.utils.lodo_cuda_policy import activate_policy, validate_runtime
import torch

from reid.data.build import build_test_loader, build_train_loader
from reid.engine.train_loop import train_one_epoch
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.optim.build import build_center_optimizer, build_optimizer, build_scheduler
from reid.optim.lr_scheduler import step_completed_epoch
from reid.utils.artifacts import save_run_artifacts
from reid.utils.checkpoint import load_checkpoint, save_checkpoint
from reid.utils.config import model_requires_num_classes, validate_model_loss_requirements
from reid.utils.config import load_config, save_yaml, validate_reid_config
from reid.utils.config_schema import validate_config
from reid.utils.device import device_summary, select_device
from reid.utils.io import ensure_dir
from reid.utils.logger import setup_logger
from reid.utils.metrics_artifacts import (
    build_metric_payload,
    save_final_epoch_metrics,
    save_final_eval_metrics,
    save_train_eval_metrics,
)
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
    return str(REPO_ROOT / path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "-o",
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
    artifacts_dir = osp.join(exp_dir, "artifacts")
    tensorboard_dir = osp.join(exp_dir, "tensorboard")
    ensure_dir(metrics_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(tensorboard_dir)
    tb_compat_dir = osp.join(exp_dir, "tb")
    if not osp.exists(tb_compat_dir):
        try:
            os.symlink("tensorboard", tb_compat_dir)
        except OSError:
            pass
    return metrics_dir, checkpoints_dir, logs_dir, plots_dir


def setup_tensorboard(cfg, exp_dir):
    tb = None
    if cfg["logging"]["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = osp.join(exp_dir, "tensorboard")
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
    if scheduler is not None and getattr(scheduler, 'step_interval', 'batch') == 'epoch':
        scheduler.validate_completed_epoch(checkpoint.get('epoch'))
    scores = checkpoint.get("scores")
    best_metric = -1.0
    metric_name = cfg["train"]["save"]["metric"]
    if isinstance(scores, dict) and metric_name in scores:
        best_metric = float(scores[metric_name])
    logger.info("Resumed from epoch %d", int(checkpoint.get("epoch", 0)))
    return start_epoch, best_metric, checkpoint


def build_eval_payload(cfg, scores, epoch=None, checkpoint_name=""):
    return build_metric_payload(cfg, scores, epoch=epoch, checkpoint_name=checkpoint_name)


def save_eval_metrics(metrics_dir, cfg, epoch, scores, checkpoint_name=""):
    return save_train_eval_metrics(metrics_dir, cfg, epoch, scores, checkpoint_name=checkpoint_name)


def is_better_score(scores, metric_name: str, best_metric: float) -> bool:
    if metric_name not in scores:
        raise KeyError(f"Configured checkpoint metric '{metric_name}' not found in eval scores.")
    return float(scores[metric_name]) > float(best_metric)


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



def run_lodo_training(cfg):
    """Target-free shared-engine execution; optional retained boundary recovery."""
    from datetime import datetime, timezone
    import shutil
    import time
    from reid.engine.update_control import UpdateProgress
    from reid.utils.lodo_protocol import validate_output_path
    from reid.utils.lodo_checkpoint import (atomic_write, canonical_json, loader_contract,
        make_identity, save_lodo_checkpoint, restore_lodo_checkpoint, validate_resume_path, read_checkpoint)
    from reid.utils.lodo_device import authorization_uuid, select_lodo_device
    from reid.utils.lodo_recovery import (recovery_policy, cuda_boundary, select_generation,
                                         recovery_event_environment)

    cfg = copy.deepcopy(cfg)
    authorization_uuid(cfg)
    activate_policy(cfg)  # opt-in only, before device discovery, seeding or loaders
    policy = recovery_policy(cfg)
    run_dir = validate_output_path(cfg)
    resume = cfg['train']['save'].get('resume', '')
    checkpoint_dir = run_dir / 'checkpoints'
    last = checkpoint_dir / 'ckpt_last.pth'
    terminal = checkpoint_dir / 'ckpt_terminal.pth'
    path = validate_resume_path(cfg, run_dir) if resume else None
    fallback = []
    if resume and policy is not None and Path(resume).name != 'ckpt_terminal.pth':
        path, fallback = select_generation(cfg, run_dir)
    if resume and (terminal.exists() or terminal.with_suffix('.json').exists()):
        path = terminal
    inspect_first = cuda_boundary(cfg) and bool(resume)
    if inspect_first:
        device, device_guard, device_evidence = torch.device('cpu'), None, None
    else:
        device, device_guard, device_evidence = select_lodo_device(cfg)
    set_seed(seed=cfg['repro']['seed'], deterministic=cfg['repro']['deterministic'],
             benchmark=cfg['repro']['benchmark'])
    validate_runtime(cfg)
    loader, classes = build_train_loader(cfg)
    loader_contract(loader)
    validate_reid_config(cfg, num_classes=classes)
    model = build_model(cfg, num_classes=classes, initialize_pretrained=not bool(resume)).to(device)
    dim = validate_model_loss_requirements(cfg, model)
    criterion = build_criterion(cfg, num_classes=classes, feat_dim=dim).to(device)
    optimizer = build_optimizer(cfg, model)
    auxiliary = build_center_optimizer(cfg, criterion)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(loader))
    arguments = dict(cfg=cfg, loader=loader, model=model, criterion=criterion,
                     optimizer=optimizer, auxiliary=auxiliary, scheduler=scheduler)
    make_identity(**arguments, cpu_inspection=inspect_first)
    if inspect_first:
        # All tensor/provenance/worker/position checks on CPU BEFORE CUDA discovery.
        checked, _, checked_kind = restore_lodo_checkpoint(path, **arguments,
                                                           cpu_inspection=True, validate_only=True)
        if checked_kind == 'terminal':
            return checked
        device, device_guard, device_evidence = select_lodo_device(cfg)
        model.to(device);criterion.to(device)
    events = []
    event_path = run_dir / 'recovery_events.json'
    if resume:
        progress, position, kind = restore_lodo_checkpoint(path, **arguments)
        epoch = position['next_epoch']
        if kind == 'terminal':
            return progress
        last = path
        events = json.loads(event_path.read_text()) if event_path.exists() else []
        event = dict(event='rollback_resume', checkpoint=str(path),
                     restored_primary_updates=progress.completed_primary_updates,
                     restart_epoch=epoch, discarded_epoch_replayed=True)
        if policy is not None:
            _, proof = read_checkpoint(path)
            event.update(recovery_event_environment(cfg),checkpoint_sha256=proof['checkpoint_sha256'],
                         invalid_generations=fallback, cuda_admission=device_evidence)
        events.append(event)
        atomic_write(event_path, lambda stream: stream.write(canonical_json(events)))
    else:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
        checkpoint_dir.mkdir()
        if device_evidence is not None:
            atomic_write(run_dir / 'cuda_admission.json',
                         lambda stream: stream.write(canonical_json(device_evidence)))
        progress = UpdateProgress(cfg['train']['max_updates'], auxiliary=auxiliary is not None)
        progress.bind(loader)
        epoch = 1

    def save_boundary(destination, *, terminal_snapshot=False, cursor=0):
        started = time.perf_counter()
        proof = save_lodo_checkpoint(destination, progress=progress, next_epoch=epoch,
                     cursor=cursor, kind='terminal' if terminal_snapshot else 'recovery', **arguments)
        duration = time.perf_counter()-started
        actual = destination
        if policy is not None:
            if not terminal_snapshot:
                selection = copy.deepcopy(cfg);selection['train']['save']['resume'] = str(checkpoint_dir/'ckpt_last.pth')
                actual, _ = select_generation(selection, run_dir)
            metrics_path = run_dir/'checkpoint_metrics.json'
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
            metrics.append(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(actual),
                                kind='terminal' if terminal_snapshot else 'recovery',
                                completed_updates=progress.completed_primary_updates,write_seconds=duration,
                                bytes=actual.stat().st_size,disk_free_bytes=shutil.disk_usage(run_dir).free,
                                checkpoint_sha256=proof['checkpoint_sha256']))
            atomic_write(metrics_path,lambda stream:stream.write(canonical_json(metrics)))
        return actual

    if not resume:
        last = save_boundary(last)
    boundary_updates = progress.completed_primary_updates
    boundary_attempts = progress.attempted_batches
    boundary_exposure = copy.deepcopy(progress.exposure.presentations)
    try:
        while not progress.budget_completed:
            if device_guard is not None:
                device_guard()
            result = train_one_epoch(model, loader, criterion, optimizer, auxiliary, device,
                                     False, cfg['system']['log_interval'], scheduler=scheduler,
                                     epoch=epoch, update_progress=progress, before_update=device_guard)
            if progress.budget_completed:
                if device_guard is not None:
                    device_guard()
                save_boundary(terminal, terminal_snapshot=True,cursor=result.attempted_batches)
                return progress
            epoch += 1
            if policy is None or (epoch-1) % policy['every_epochs'] == 0:
                last = save_boundary(checkpoint_dir/'ckpt_last.pth')
                boundary_updates = progress.completed_primary_updates
                boundary_attempts = progress.attempted_batches
                boundary_exposure = copy.deepcopy(progress.exposure.presentations)
    except BaseException as error:
        recoverable = device.type == 'cpu' or cuda_boundary(cfg)
        event = dict(event='interruption_rollback_required' if recoverable else 'interruption_fresh_restart_required',
                     epoch=epoch, discarded_attempted_batches=progress.attempted_batches-boundary_attempts,
                     discarded_successful_updates=progress.completed_primary_updates-boundary_updates,
                     discarded_source_presentations={phase: dict(counts-boundary_exposure[phase])
                         for phase,counts in progress.exposure.presentations.items()},
                     consistent=progress.consistent, reason=str(error), recover_from=str(last) if recoverable else None)
        if policy is not None:
            event.update(recovery_event_environment(cfg),source_boundary_updates=boundary_updates)
        events.append(event)
        atomic_write(event_path, lambda stream: stream.write(canonical_json(events)))
        raise
    return progress


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.opts)
    from reid.utils.lodo_protocol import has_lodo_protocol
    if has_lodo_protocol(cfg):
        from reid.utils.lodo_operational import validate_training_admission
        validate_training_admission(args.config, cfg, root=REPO_ROOT)
        validate_config(cfg)
        validate_reid_config(cfg)
        return run_lodo_training(cfg)
    validate_config(cfg)
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
        artifact_paths = save_run_artifacts(exp_dir, argv=sys.argv)

        device, _ = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0), cfg)
        set_seed(
            seed=int(cfg["repro"]["seed"]),
            deterministic=bool(cfg["repro"]["deterministic"]),
            benchmark=bool(cfg["repro"]["benchmark"]),
        )

        tb = setup_tensorboard(cfg, exp_dir)

        train_loader, train_num_classes = build_train_loader(cfg)
        batch_size = getattr(train_loader, "effective_batch_size", None)
        train_dataset = train_loader.dataset

        classifier_enabled = model_requires_num_classes(cfg)
        center_enabled = bool(cfg["loss"].get("center", {}).get("enabled"))
        num_classes = None
        if classifier_enabled or bool(cfg["loss"].get("id", {}).get("enabled")) or center_enabled:
            num_classes = train_num_classes
            if num_classes is None:
                raise ValueError(
                    "Classifier, ID loss, or center loss enabled but train dataset has no "
                    "'num_classes' attribute."
                )
            validate_reid_config(cfg, num_classes=num_classes)

        test_loader = build_test_loader(cfg)

        model = build_model(cfg, num_classes=num_classes).to(device)
        feat_dim = validate_model_loss_requirements(cfg, model)

        criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=feat_dim).to(device)
        optimizer = build_optimizer(cfg, model)
        center_optimizer = build_center_optimizer(cfg, criterion)
        scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

        resume_path = str(cfg["train"]["save"].get("resume", "") or "").strip()
        log_startup(logger, args, cfg, device, train_loader, num_classes, optimizer, scheduler, resume_path)
        logger.info("Train batch size: %s", batch_size)
        logger.info("Resolved config saved to: %s", resolved_config_path)
        logger.info("Command saved to: %s", artifact_paths["command"])
        logger.info("Environment saved to: %s", artifact_paths["environment"])
        if "git_commit" in artifact_paths:
            logger.info("Git commit saved to: %s", artifact_paths["git_commit"])
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
        final_scores = None
        final_eval_epoch = None

        for ep in range(start_epoch, epochs + 1):
            avg_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                aux_optimizer=center_optimizer,
                device=device,
                amp=amp,
                log_interval=log_interval,
                scheduler=scheduler,
                tb_writer=tb,
                epoch=ep,
                logger=logger,
            )
            step_completed_epoch(scheduler, ep)
            loss_history.append({"epoch": ep, "avg_loss": float(avg_loss)})
            logger.info("[Epoch %d] avg_loss=%.4f", ep, avg_loss)

            # Persist this epoch before evaluation so validation metrics that
            # identify ckpt_last.pth always refer to the evaluated model state.
            if save_last:
                last_path = checkpoint_paths["last"]
                save_checkpoint(
                    path=last_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    center_optimizer=center_optimizer,
                    epoch=ep,
                    scores=None,
                    cfg=cfg,
                )

            latest_scores = None
            if (ep % eval_interval) == 0:
                scores = evaluate_reid(cfg, model, test_loader, device, logger=logger)
                latest_scores = scores
                final_scores = scores
                final_eval_epoch = ep
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
                    if is_better_score(scores, best_name, best_metric):
                        best_metric = float(scores[best_name])
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

                # Refresh the already-written checkpoint with its evaluation
                # payload without changing the model state it represents.
                if save_last:
                    save_checkpoint(
                        path=checkpoint_paths["last"],
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        center_optimizer=center_optimizer,
                        epoch=ep,
                        scores=latest_scores,
                        cfg=cfg,
                    )

        if final_scores is not None:
            # Keep final-epoch scores separate from the benchmark result of the
            # selected checkpoint.  Previously final epoch metrics could be
            # labelled as ckpt_best.pth even when best was from an earlier epoch.
            save_final_epoch_metrics(
                metrics_dir,
                cfg,
                epoch=final_eval_epoch,
                scores=final_scores,
                checkpoint_name=osp.basename(checkpoint_paths["last"]),
            )

        if save_best and osp.isfile(checkpoint_paths["best"]):
            best_checkpoint = load_checkpoint(
                checkpoint_paths["best"], model=model, map_location=device
            )
            best_scores = evaluate_reid(cfg, model, test_loader, device, logger=logger)
            best_epoch = int(best_checkpoint.get("epoch", 0))
            save_final_eval_metrics(
                metrics_dir,
                cfg,
                epoch=best_epoch,
                scores=best_scores,
                checkpoint_name=osp.basename(checkpoint_paths["best"]),
            )
            logger.info("[Final Eval] checkpoint=ckpt_best.pth epoch=%d", best_epoch)
        elif final_scores is not None:
            # A run without a best checkpoint still produces a truthful final
            # result, explicitly attributed to the final/last checkpoint.
            save_final_eval_metrics(
                metrics_dir,
                cfg,
                epoch=final_eval_epoch,
                scores=final_scores,
                checkpoint_name=osp.basename(checkpoint_paths["last"]),
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
