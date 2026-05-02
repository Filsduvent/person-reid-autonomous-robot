import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reid.data.build import build_test_loader, build_train_loader
from reid.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test CUHK03 processed dataset loaders.")
    parser.add_argument(
        "--config",
        default="configs/baseline_cuhk03_resnet50_triplet.yaml",
        help="CUHK03 training config to load.",
    )
    parser.add_argument(
        "--root",
        default="",
        help="Optional dataset root override, e.g. /path/to/Dataset.",
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Only construct loaders and print dataset stats; do not fetch a batch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = ["data.num_workers=0"]
    if args.root:
        overrides.append(f"data.root={args.root}")

    cfg = load_config(args.config, overrides=overrides)
    train_loader, batch_size = build_train_loader(cfg)
    test_loader = build_test_loader(cfg)

    print(f"train batch size: {batch_size}")
    print(f"train dataset class: {train_loader.dataset.__class__.__name__}")
    print(f"test dataset class: {test_loader.dataset.__class__.__name__}")

    if args.skip_batch:
        return

    train_imgs, train_labels = next(iter(train_loader))
    test_imgs, pids, camids, names, marks = next(iter(test_loader))

    print(f"train batch images: {tuple(train_imgs.shape)}")
    print(f"train batch labels: {tuple(train_labels.shape)}")
    print(f"test batch images: {tuple(test_imgs.shape)}")
    print(f"test pids/camids/marks: {tuple(pids.shape)} {tuple(camids.shape)} {tuple(marks.shape)}")
    print(f"test names example: {names[0] if names else 'n/a'}")


if __name__ == "__main__":
    main()
