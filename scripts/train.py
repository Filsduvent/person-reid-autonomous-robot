import argparse
import torch

from utils.config import load_config
from utils.device import select_device, device_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = select_device(cfg["system"]["device"], cfg["system"].get("gpu_id", 0))
    print(f"[Device] {device_summary(device)}")

    # Example: set seed
    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # From here: build dataset/model/loss/optimizer using cfg
    # (Step 2 will refactor your training entrypoint cleanly.)

if __name__ == "__main__":
    main()
