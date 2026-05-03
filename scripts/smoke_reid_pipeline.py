import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reid.data.build import build_test_loader, build_train_loader
from reid.losses.build import build_criterion
from reid.models.build import build_model
from reid.models.outputs import ensure_output_dict
from reid.utils.config import load_config, validate_reid_config
from reid.utils.device import select_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test the ReID train/eval pipeline for one dataset config."
    )
    parser.add_argument("--config", required=True, help="Training config to load.")
    parser.add_argument("--root", default="", help="Optional dataset root override.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device override for the smoke forward pass. Use cpu by default.",
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Only construct loaders and print dataset stats; do not fetch batches.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model, forward, and loss checks after fetching batches.",
    )
    parser.add_argument(
        "--use-config-pretrained",
        action="store_true",
        help="Honor model.backbone.pretrained from the config. Default disables it to avoid downloads.",
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Optional config overrides as key=value pairs.",
    )
    return parser.parse_args()


def _base_overrides(args):
    overrides = ["data.num_workers=0", f"system.device={args.device}"]
    if not args.use_config_pretrained:
        overrides.append("model.backbone.pretrained=false")
    if args.root:
        overrides.append(f"data.root={args.root}")
    overrides.extend(args.opts)
    return overrides


def describe_train_batch(batch):
    imgs, labels = batch
    if not torch.is_tensor(imgs) or imgs.ndim != 4:
        raise ValueError("Train batch images must be an NCHW tensor.")
    if not torch.is_tensor(labels) or labels.ndim != 1:
        raise ValueError("Train batch labels must be a 1D tensor.")
    if imgs.shape[0] != labels.shape[0]:
        raise ValueError("Train batch image count must match label count.")
    return {
        "images": tuple(imgs.shape),
        "labels": tuple(labels.shape),
        "dtype": str(labels.dtype),
    }


def describe_eval_batch(batch):
    imgs, pids, camids, names, marks = batch
    if not torch.is_tensor(imgs) or imgs.ndim != 4:
        raise ValueError("Eval batch images must be an NCHW tensor.")
    for field_name, values in (("pids", pids), ("camids", camids), ("marks", marks)):
        if not torch.is_tensor(values) or values.ndim != 1:
            raise ValueError(f"Eval batch {field_name} must be a 1D tensor.")
        if values.shape[0] != imgs.shape[0]:
            raise ValueError(f"Eval batch {field_name} count must match image count.")
    if len(names) != imgs.shape[0]:
        raise ValueError("Eval batch names count must match image count.")
    return {
        "images": tuple(imgs.shape),
        "pids": tuple(pids.shape),
        "camids": tuple(camids.shape),
        "marks": tuple(marks.shape),
        "first_name": names[0] if names else "n/a",
    }


def run_smoke(args):
    cfg = load_config(args.config, overrides=_base_overrides(args))
    validate_reid_config(cfg)

    train_loader, batch_size = build_train_loader(cfg)
    test_loader = build_test_loader(cfg)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    num_classes = int(getattr(train_dataset, "num_classes"))

    print(f"config: {args.config}")
    print(f"train dataset: {train_dataset.__class__.__name__}")
    print(f"test dataset: {test_dataset.__class__.__name__}")
    print(f"num_classes: {num_classes}")
    print(f"train batch size: {batch_size}")

    if args.skip_batch:
        print("batch checks: skipped")
        return

    train_batch = next(iter(train_loader))
    eval_batch = next(iter(test_loader))
    train_info = describe_train_batch(train_batch)
    eval_info = describe_eval_batch(eval_batch)

    print(f"train batch images: {train_info['images']}")
    print(f"train batch labels: {train_info['labels']} {train_info['dtype']}")
    print(f"eval batch images: {eval_info['images']}")
    print(f"eval pids/camids/marks: {eval_info['pids']} {eval_info['camids']} {eval_info['marks']}")
    print(f"eval first name: {eval_info['first_name']}")

    if args.skip_model:
        print("model/loss checks: skipped")
        return

    validate_reid_config(cfg, num_classes=num_classes)
    device, _ = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0), cfg)
    model = build_model(cfg, num_classes=num_classes).to(device)
    criterion = build_criterion(cfg, num_classes=num_classes, feat_dim=model.feat_dim).to(device)

    model.train()
    train_imgs, train_labels = train_batch
    train_imgs = train_imgs.to(device)
    train_labels = train_labels.to(device)
    outputs = ensure_output_dict(model(train_imgs))
    loss, loss_logs = criterion(outputs, train_labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss is not finite.")
    print(f"forward emb: {tuple(outputs['emb'].shape)}")
    print(f"loss total: {loss_logs['loss/total']:.6f}")

    model.eval()
    eval_imgs = eval_batch[0].to(device)
    with torch.no_grad():
        eval_outputs = ensure_output_dict(model(eval_imgs))
    print(f"eval forward emb: {tuple(eval_outputs['emb'].shape)}")


def main():
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
