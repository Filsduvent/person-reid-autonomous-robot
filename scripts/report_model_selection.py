"""Build a reproducible architecture-selection report from cross-dataset JSONs."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reid.utils.experiment_matrix import (
    build_model_selection_report,
    load_cross_dataset_records,
    write_model_selection_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", required=True, help="Glob for cross_dataset.json files.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()
    paths = sorted(glob.glob(args.results_glob, recursive=True))
    if not paths:
        raise SystemExit("No cross-dataset result files matched --results-glob.")
    report = build_model_selection_report(load_cross_dataset_records(paths))
    write_model_selection_report(args.output_json, report, args.output_csv or None)
    print(f"selected_trained_reid_model={report['selected_trained_reid_model']}")


if __name__ == "__main__":
    main()
