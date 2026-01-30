#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def norm(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "model": row.get("backbone", ""),
        "device": row.get("device", ""),
        "weights_path": row.get("weights", ""),
        "model_size_mb": row.get("model_size_mb", ""),
        "param_count": row.get("param_count", ""),
        "fps": row.get("fps", ""),
        "latency_ms_mean": row.get("total_ms_mean", ""),
        "latency_ms_p95": row.get("total_ms_p95", ""),
        "rss_mb_mean": row.get("rss_mb_mean", ""),
        "rss_mb_p95": row.get("rss_mb_p95", ""),
        "vram_mb_mean": row.get("vram_mb_mean", ""),
        "vram_mb_p95": row.get("vram_mb_p95", ""),
        "detector_ms_mean": row.get("detector_ms_mean", ""),
        "tracker_ms_mean": row.get("tracker_ms_mean", ""),
        "embedder_ms_mean": row.get("embedder_ms_mean", ""),
        "gallery_ms_mean": row.get("gallery_ms_mean", ""),
        "visualization_ms_mean": row.get("visualization_ms_mean", ""),
        "input_ms_mean": row.get("input_ms_mean", ""),
        "total_ms_mean": row.get("total_ms_mean", ""),
        "notes": row.get("name", ""),
    }


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        die("Usage: scripts/aggregate_results.py outputs/benchmarks/summary.csv results/final_results.csv")
    in_path = Path(argv[1])
    out_path = Path(argv[2])
    if not in_path.exists():
        die(f"Input summary not found: {in_path}")

    rows = read_summary(in_path)
    if not rows:
        die("No rows found in input summary.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "device",
        "weights_path",
        "model_size_mb",
        "param_count",
        "fps",
        "latency_ms_mean",
        "latency_ms_p95",
        "rss_mb_mean",
        "rss_mb_p95",
        "vram_mb_mean",
        "vram_mb_p95",
        "detector_ms_mean",
        "tracker_ms_mean",
        "embedder_ms_mean",
        "gallery_ms_mean",
        "visualization_ms_mean",
        "input_ms_mean",
        "total_ms_mean",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(norm(row))

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
