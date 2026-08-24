"""Re-evaluate completed checkpoints after a retrieval-metric correction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def completed_runs(selected_runs=()):
    selected_runs = {Path(run).as_posix() for run in selected_runs}
    for metric_path in sorted((REPO_ROOT / "exp").glob("*/metrics/final_test.json")):
        run_dir = metric_path.parents[1]
        if (run_dir / "config.yaml").is_file() and (run_dir / "checkpoints/ckpt_best.pth").is_file():
            if selected_runs and run_dir.relative_to(REPO_ROOT).as_posix() not in selected_runs:
                continue
            yield run_dir


def main():
    selected_runs = sys.argv[1:]
    for run_dir in completed_runs(selected_runs):
        config_path = run_dir / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset = cfg["data"]["test"]["dataset"]["name"]
        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--config", str(config_path.relative_to(REPO_ROOT)),
            "--weight", str((run_dir / "checkpoints/ckpt_best.pth").relative_to(REPO_ROOT)),
        ]
        if dataset == "cuhk03":
            command += [
                "--opts",
                "data.test.dataset.protocol=processed_partition",
                "data.test.dataset.split_id=null",
            ]
        log_path = run_dir / "logs/corrected_minp_evaluate.log"
        print(f"[START] {run_dir.relative_to(REPO_ROOT)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        print(f"[DONE] {run_dir.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
