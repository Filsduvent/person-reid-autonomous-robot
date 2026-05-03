import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIGS = {
    "market1501": "configs/baseline_market1501_resnet50_triplet.yaml",
    "duke": "configs/baseline_duke_resnet50_triplet.yaml",
    "cuhk03": "configs/baseline_cuhk03_resnet50_triplet.yaml",
    "msmt17": "configs/baseline_msmt17_resnet50_triplet.yaml",
}


SMOKE_OVERRIDES = [
    "system.device=cpu",
    "system.amp=false",
    "system.log_interval=1",
    "data.num_workers=0",
    "data.pin_memory=false",
    "model.backbone.pretrained=false",
    "train.epochs=1",
    "train.eval_interval=1",
    "data.train.batch.P=4",
    "data.train.batch.K=2",
    "data.train.batch.batch_size=8",
    "data.test.batch.size=8",
    "logging.tensorboard=false",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the one-epoch ReID smoke-test matrix for supported datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS),
        choices=sorted(DATASET_CONFIGS),
        help="Dataset smoke tests to run.",
    )
    parser.add_argument("--root", default="", help="Optional dataset root override.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device override passed to system.device. Defaults to CPU for portability.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print train.py commands without executing them.",
    )
    parser.add_argument(
        "-o",
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra key=value overrides appended after the default smoke overrides.",
    )
    return parser.parse_args()


def build_smoke_command(dataset_name, args):
    config_path = DATASET_CONFIGS[dataset_name]
    experiment_name = f"smoke_{dataset_name}_resnet50"
    overrides = list(SMOKE_OVERRIDES)
    overrides[0] = f"system.device={args.device}"
    overrides.extend([
        f"experiment.name={experiment_name}",
        f"experiment.output_dir=exp/{experiment_name}",
    ])
    if args.root:
        overrides.append(f"data.root={args.root}")
    overrides.extend(args.opts)

    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config",
        str(REPO_ROOT / config_path),
        "-o",
        *overrides,
    ]


def run_matrix(args):
    for dataset_name in args.datasets:
        cmd = build_smoke_command(dataset_name, args)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    run_matrix(parse_args())


if __name__ == "__main__":
    main()
