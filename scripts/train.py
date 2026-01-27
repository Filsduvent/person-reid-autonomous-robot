import argparse
import json, os.path as osp
import torch
from torch.utils.data import DataLoader

from reid.utils.config import load_config
from reid.utils.device import select_device, device_summary
from reid.utils.io import ensure_dir
from reid.utils.seed import set_seed

from reid.data.build import build_train_loader
from reid.data.market1501_test import Market1501TestFromPartitions
from reid.data.transforms import build_test_tf
from reid.models.build import build_model
from reid.losses.build import build_criterion
from reid.engine.evaluator import evaluate_reid
from reid.engine.train_loop import train_one_epoch

def build_optimizer(cfg, model):
    ocfg = cfg["optim"]
    name = ocfg["name"].lower()
    lr = float(ocfg["lr"])
    wd = float(ocfg["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_dir = cfg["experiment"]["output_dir"]
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

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    optimizer = build_optimizer(cfg, model)

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

    # Step 2.1: keep epoch length bounded (later we make it precise)
    steps_per_epoch = 200

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
            device=device,
            amp=amp,
            log_interval=log_interval,
            tb_writer=tb,
            epoch=ep,
            steps_per_epoch=steps_per_epoch,
        )
        print(f"[Epoch {ep}] avg_loss={avg_loss:.4f}")

        if (ep % int(cfg["train"]["eval_interval"])) == 0:
            scores = evaluate_reid(cfg, model, test_loader, device)
            print(f"[Eval] epoch={ep} mAP={scores['mAP']:.4f} Rank1={scores['Rank1']:.4f} mINP={scores['mINP']:.4f}")

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
