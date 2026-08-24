"""Portable cross-dataset result records and model-selection summaries."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


METRIC_KEYS = ("mAP", "mINP", "Rank1", "Rank5", "Rank10")


def build_cross_dataset_record(*, source_dataset, target_dataset, architecture, checkpoint_path, cfg, scores):
    """Build the machine-readable result for one source-to-target evaluation."""
    record = {
        "schema_version": 1,
        "experiment_id": f"{architecture}_{source_dataset}_to_{target_dataset}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "resolved_config": cfg,
    }
    for key in METRIC_KEYS:
        record[key] = float(scores[key]) if scores.get(key) is not None else None
    return record


def write_cross_dataset_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_cross_dataset_records(paths):
    records = []
    for path in paths:
        records.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return records


def _rank_condition(rows):
    """Rank by mAP, with Rank-1 as the deterministic tie-breaker."""
    ordered = sorted(rows, key=lambda row: (-float(row["mAP"]), -float(row["Rank1"])))
    return {row["architecture"]: index + 1 for index, row in enumerate(ordered)}


def build_model_selection_report(records, datasets=("market1501", "duke", "cuhk03", "msmt17")):
    """Return ranks only for architectures with a complete 4x4 result matrix."""
    datasets = tuple(datasets)
    architectures = sorted({row["architecture"] for row in records})
    latest = {}
    for row in records:
        key = (row["architecture"], row["source_dataset"], row["target_dataset"])
        latest[key] = row

    expected = [(source, target) for source in datasets for target in datasets]
    complete = []
    summary = []
    for architecture in architectures:
        missing = [f"{source}->{target}" for source, target in expected if (architecture, source, target) not in latest]
        item = {
            "architecture": architecture,
            "mean_within_domain_mAP_rank": None,
            "mean_cross_domain_mAP_rank": None,
            "selection_score": None,
            "mean_Rank1": None,
            "missing_experiments": missing,
        }
        if not missing:
            complete.append(architecture)
        summary.append(item)

    condition_ranks = {}
    for source, target in expected:
        rows = [latest[(architecture, source, target)] for architecture in complete]
        if rows:
            condition_ranks[(source, target)] = _rank_condition(rows)

    for item in summary:
        architecture = item["architecture"]
        if architecture not in complete:
            continue
        within = [condition_ranks[(dataset, dataset)][architecture] for dataset in datasets]
        cross = [
            condition_ranks[(source, target)][architecture]
            for source, target in expected
            if source != target
        ]
        rows = [latest[(architecture, source, target)] for source, target in expected]
        item["mean_within_domain_mAP_rank"] = sum(within) / len(within)
        item["mean_cross_domain_mAP_rank"] = sum(cross) / len(cross)
        item["selection_score"] = (
            item["mean_within_domain_mAP_rank"] + item["mean_cross_domain_mAP_rank"]
        ) / 2.0
        item["mean_Rank1"] = sum(float(row["Rank1"]) for row in rows) / len(rows)

    eligible = [item for item in summary if item["selection_score"] is not None]
    selected = None
    # Model selection is intentionally deferred until every architecture under
    # comparison has a complete matrix; partial results must not crown a model.
    if eligible and len(complete) == len(architectures):
        eligible.sort(key=lambda item: (item["selection_score"], -item["mean_Rank1"]))
        selected = eligible[0]["architecture"]
    return {"datasets": list(datasets), "architectures": summary, "selected_trained_reid_model": selected}


def write_model_selection_report(json_path, report, csv_path=None):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "architecture", "mean_within_domain_mAP_rank", "mean_cross_domain_mAP_rank",
                "selection_score", "mean_Rank1", "missing_experiments",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report["architectures"]:
                row = dict(row)
                row["missing_experiments"] = ";".join(row["missing_experiments"])
                writer.writerow(row)
