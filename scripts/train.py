# scripts/train.py
import argparse
import os
import torch

from reid.utils.config import load_config, save_yaml
from reid.utils.device import select_device, device_summary


def set_seed(seed: int, device: torch.device, deterministic: bool = False, benchmark: bool = True):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # reproducibility flags
    torch.backends.cudnn.benchmark = bool(benchmark) if device.type == "cuda" else False
    torch.backends.cudnn.deterministic = bool(deterministic) if device.type == "cuda" else False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "--override", "-o",
        action="append",
        default=[],
        help="Dotpath overrides. Example: -o system.device=cuda -o train.epochs=60"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)

    # Backward compatibility: if user still has experiment.seed, map it into repro.seed if missing
    if "repro" not in cfg:
        cfg["repro"] = {}
    if "seed" not in cfg["repro"]:
        cfg["repro"]["seed"] = cfg.get("experiment", {}).get("seed", 42)

    sys_cfg = cfg.get("system", {})
    device_str = sys_cfg.get("device", "auto")
    gpu_id = int(sys_cfg.get("gpu_id", 0))

    device, cfg = select_device(device_str, gpu_id, cfg=cfg)
    print(f"[Device] {device_summary(device)}")

    # Ensure output dir exists
    exp = cfg.get("experiment", {})
    out_dir = exp.get("output_dir", "exp/run")
    os.makedirs(out_dir, exist_ok=True)

    # Save resolved config (after device policy)
    save_yaml(cfg, os.path.join(out_dir, "config.resolved.yaml"))

    # Seed + deterministic flags
    repro = cfg.get("repro", {})
    seed = int(repro.get("seed", 42))
    deterministic = bool(repro.get("deterministic", False))
    benchmark = bool(repro.get("benchmark", True))
    set_seed(seed, device, deterministic=deterministic, benchmark=benchmark)
    print(f"[Repro] seed={seed} deterministic={deterministic} benchmark={benchmark}")

    # From here: build dataset/model/loss/optimizer using cfg (Phase A Step 2+)
    # - data loaders should use cfg["data"]["num_workers"], cfg["data"]["pin_memory"]
    # - amp should follow cfg["system"]["amp"] (already forced off on CPU)


if __name__ == "__main__":
    main()
