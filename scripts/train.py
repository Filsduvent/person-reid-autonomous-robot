import argparse
import json, os.path as osp
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from reid.utils.config import load_config, validate_reid_config
from reid.utils.device import select_device, device_summary
from reid.utils.io import ensure_dir
from reid.utils.seed import set_seed

from reid.data.build import build_train_loader
from reid.data.market1501_test import Market1501TestFromPartitions
from reid.data.transforms import build_test_tf
from reid.models.build import build_model
from reid.losses.build import build_criterion
from reid.optim.build import build_optimizer, build_center_optimizer, build_scheduler
from reid.engine.evaluator import evaluate_reid
from reid.engine.train_loop import train_one_epoch


def resolve_repo_relative_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / path)


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_reid_config(cfg)

    exp_dir = resolve_repo_relative_path(cfg["experiment"]["output_dir"])
    cfg["experiment"]["output_dir"] = exp_dir
    ensure_dir(exp_dir)

    device, _ = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0), cfg)
    print(f"[Device] {device_summary(device)}")

    set_seed(
        seed=int(cfg["repro"]["seed"]),
        deterministic=bool(cfg["repro"]["deterministic"]),
        benchmark=bool(cfg["repro"]["benchmark"]),
    )

    # TensorBoard (optional)
    tb = None
    if cfg["logging"]["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = osp.join(exp_dir, "tb")
        ensure_dir(tb_dir)
        tb = SummaryWriter(tb_dir)

    train_loader, batch_size = build_train_loader(cfg)
    print(f"[Data] Train batch size = {batch_size}")

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

    model = build_model(cfg, num_classes=num_classes).to(device)
    feat_dim = getattr(model, "feat_dim", None)
    if feat_dim is None or int(feat_dim) <= 0:
        raise ValueError("Model must expose a positive 'feat_dim' attribute for loss construction.")
    criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=feat_dim).to(device)
    optimizer = build_optimizer(cfg, model)
    center_optimizer = build_center_optimizer(cfg, criterion)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    # --- Test loader ---
    tcfg = cfg["data"]["test"]
    root = cfg["data"]["root"]
    image_size = tuple(tcfg["images"]["size"])
    aug = tcfg["aug"]
    test_tf = build_test_tf(image_size=image_size, mean=aug["mean"], std=aug["std"])

    if tcfg["dataset"]["name"] != "market1501":
        raise NotImplementedError("This step wires market1501 test first.")

    test_ds = Market1501TestFromPartitions(root=root, transform=test_tf, split=tcfg["dataset"]["split"])
    test_loader = DataLoader(
        test_ds,
        batch_size=int(tcfg["batch"]["size"]),
        shuffle=bool(tcfg["loader"]["shuffle"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )

    amp = bool(cfg["system"]["amp"])
    log_interval = int(cfg["system"]["log_interval"])
    epochs = int(cfg["train"]["epochs"])

    best_metric = -1.0
    best_name = cfg["train"]["save"]["metric"]  # "mAP" recommended

    metrics_dir = osp.join(exp_dir, "metrics")
    ensure_dir(metrics_dir)

    for ep in range(1, epochs + 1):
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
        )
        print(f"[Epoch {ep}] avg_loss={avg_loss:.4f}")

        if (ep % int(cfg["train"]["eval_interval"])) == 0:
            scores = evaluate_reid(cfg, model, test_loader, device)
            print(f"[Eval] epoch={ep} mAP={scores['mAP']:.4f} Rank1={scores['Rank1']:.4f} mINP={scores['mINP']:.4f}")
            log_eval_metrics(tb, scores, ep)

            # save metrics
            with open(osp.join(metrics_dir, "latest_val.json"), "w") as f:
                json.dump({"epoch": ep, **scores}, f, indent=2)
            with open(osp.join(metrics_dir, f"val_epoch_{ep:03d}.json"), "w") as f:
                json.dump({"epoch": ep, **scores}, f, indent=2)

            # best ckpt
            cur = float(scores[best_name])
            if cur > best_metric:
                best_metric = cur
                best_path = osp.join(exp_dir, "ckpt_best.pth")
                torch.save({"epoch": ep, "model": model.state_dict(), "optim": optimizer.state_dict(), "scores": scores}, best_path)
                print(f"[CKPT] New best {best_name}={best_metric:.4f} saved: {best_path}")

        # always save last
        ckpt_path = osp.join(exp_dir, "ckpt_last.pth")
        torch.save({"epoch": ep, "model": model.state_dict(), "optim": optimizer.state_dict()}, ckpt_path)

    if tb is not None:
        tb.close()

if __name__ == "__main__":
    main()
