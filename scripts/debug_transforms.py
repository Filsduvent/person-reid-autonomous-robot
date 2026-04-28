import argparse
import os.path as osp
from pathlib import Path

import torch
from torchvision.utils import save_image

from reid.data.build import build_train_loader
from reid.utils.config import load_config, validate_reid_config
from reid.utils.io import ensure_dir
from reid.utils.seed import set_seed


def resolve_repo_relative_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / path)


def denormalize_image(img: torch.Tensor, mean, std) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    img = img * std_tensor + mean_tensor
    return img.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_reid_config(cfg)

    exp_dir = resolve_repo_relative_path(cfg["experiment"]["output_dir"])
    cfg["experiment"]["output_dir"] = exp_dir
    ensure_dir(exp_dir)

    set_seed(
        seed=int(cfg["repro"]["seed"]),
        deterministic=bool(cfg["repro"]["deterministic"]),
        benchmark=bool(cfg["repro"]["benchmark"]),
    )

    debug_dir = osp.join(exp_dir, "debug_aug")
    ensure_dir(debug_dir)

    train_loader, _ = build_train_loader(cfg)
    mean = cfg["data"]["train"]["aug"]["mean"]
    std = cfg["data"]["train"]["aug"]["std"]

    saved = 0
    for imgs, labels in train_loader:
        batch_size = imgs.size(0)
        for idx in range(batch_size):
            img = denormalize_image(imgs[idx].detach().cpu(), mean=mean, std=std)
            label = int(labels[idx])
            out_path = osp.join(debug_dir, f"sample_{saved:03d}_label_{label:04d}.png")
            save_image(img, out_path)
            saved += 1
            if saved >= args.num_samples:
                print(f"[DebugAug] Saved {saved} images to {debug_dir}")
                return

    print(f"[DebugAug] Saved {saved} images to {debug_dir}")


if __name__ == "__main__":
    main()
