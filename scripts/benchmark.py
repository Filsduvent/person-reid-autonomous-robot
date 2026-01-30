#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        die("PyYAML not installed. Install with: pip install pyyaml")
    if not path.exists():
        die(f"Benchmark config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        die("Benchmark config must be a YAML mapping.")
    return data


def summarize_series(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    p95_idx = int(0.95 * (n - 1))
    p95 = vals[p95_idx]
    return {"mean": mean, "median": median, "p95": p95}


def read_frames(path: Path) -> List[Dict[str, Any]]:
    frames = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def model_size_mb(weights_path: str | None) -> float:
    if not weights_path:
        return 0.0
    p = Path(weights_path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024.0 * 1024.0)


def param_count(weights_path: str | None) -> int | None:
    if not weights_path:
        return None
    try:
        import torch
    except Exception:
        return None
    p = Path(weights_path)
    if not p.exists():
        return None
    state = torch.load(p, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None
    count = 0
    for v in state.values():
        try:
            count += v.numel()
        except Exception:
            continue
    return count


def run_one(cfg_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    cmd = [sys.executable, "-m", "edge_reid_runtime.run", "--config", str(cfg_path)]
    return subprocess.call(cmd, env=env)


def build_run_config(base: Dict[str, Any], run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(run_cfg)
    return merged


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        die("Usage: scripts/benchmark.py configs/benchmark.yaml")
    config_path = Path(argv[1])
    data = load_yaml(config_path)

    common = data.get("common", {})
    runs = data.get("runs", [])
    output_root = Path(common.get("output_root", "outputs/benchmarks"))
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.csv"
    write_header = not summary_path.exists()

    for run in runs:
        name = run.get("name")
        if not name:
            die("Each run must include a 'name'.")
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        merged = build_run_config(common, run)
        merged["output_dir"] = str(run_dir)
        merged["save_video"] = False
        merged["display"] = False
        warmup = int(merged.pop("warmup_frames", 0))

        cfg_path = run_dir / "run.yaml"
        if yaml is None:
            die("PyYAML not installed. Install with: pip install pyyaml")
        cfg_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")

        rc = run_one(cfg_path)
        if rc != 0:
            print(f"Run failed: {name}", file=sys.stderr)
            continue

        frames_path = run_dir / "frames.jsonl"
        if not frames_path.exists():
            print(f"No frames.jsonl for run {name}", file=sys.stderr)
            continue
        frames = read_frames(frames_path)
        if warmup > 0:
            frames = frames[warmup:]

        dt_ms = [f.get("dt_ms", 0.0) for f in frames]
        rss = [f.get("rss_mb", 0.0) for f in frames if f.get("rss_mb") is not None]
        vram = [f.get("vram_mb", 0.0) for f in frames if f.get("vram_mb") is not None]

        stages = {}
        for f in frames:
            smap = f.get("stages_ms", {})
            for k, v in smap.items():
                stages.setdefault(k, []).append(float(v))

        total_s = sum(dt_ms) / 1000.0 if dt_ms else 0.0
        fps = (len(dt_ms) / total_s) if total_s > 0 else 0.0

        weights_path = merged.get("weights")
        row = {
            "name": name,
            "device": merged.get("device", "cpu"),
            "backbone": merged.get("reid_backbone", ""),
            "model_size_mb": f"{model_size_mb(weights_path):.2f}",
            "param_count": param_count(weights_path) or "",
            "frames": len(dt_ms),
            "fps": f"{fps:.2f}",
            "total_ms_mean": f"{summarize_series(dt_ms)['mean']:.2f}",
            "total_ms_p95": f"{summarize_series(dt_ms)['p95']:.2f}",
            "rss_mb_mean": f"{summarize_series(rss)['mean']:.2f}" if rss else "",
            "rss_mb_p95": f"{summarize_series(rss)['p95']:.2f}" if rss else "",
            "vram_mb_mean": f"{summarize_series(vram)['mean']:.2f}" if vram else "",
            "vram_mb_p95": f"{summarize_series(vram)['p95']:.2f}" if vram else "",
            "id_metrics": "N/A",
        }

        for stage_name in ("detector", "tracker", "embedder", "gallery", "visualization", "input", "total"):
            if stage_name in stages:
                stats = summarize_series(stages[stage_name])
                row[f"{stage_name}_ms_mean"] = f"{stats['mean']:.2f}"
                row[f"{stage_name}_ms_p95"] = f"{stats['p95']:.2f}"

        with summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)

        print(f"Completed: {name}")

    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
