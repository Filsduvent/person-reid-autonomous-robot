from reid.utils.experiment_matrix import build_model_selection_report
from scripts.evaluate_cross_domain import build_cross_domain_config


DATASETS = ("market1501", "duke", "cuhk03", "msmt17")


def _record(architecture, source, target, map_score, rank1):
    return {
        "architecture": architecture,
        "source_dataset": source,
        "target_dataset": target,
        "mAP": map_score,
        "Rank1": rank1,
    }


def test_selection_report_excludes_incomplete_architectures():
    records = []
    for source in DATASETS:
        for target in DATASETS:
            records.append(_record("complete", source, target, 0.7, 0.8))
    records.append(_record("incomplete", "market1501", "market1501", 0.9, 0.9))

    report = build_model_selection_report(records)
    rows = {row["architecture"]: row for row in report["architectures"]}
    assert report["selected_trained_reid_model"] is None
    assert rows["complete"]["selection_score"] == 1.0
    assert rows["incomplete"]["selection_score"] is None
    assert len(rows["incomplete"]["missing_experiments"]) == 15


def test_selection_tie_breaks_map_rank_with_rank1():
    records = []
    for source in DATASETS:
        for target in DATASETS:
            records.extend([
                _record("higher_rank1", source, target, 0.7, 0.9),
                _record("lower_rank1", source, target, 0.7, 0.8),
            ])
    report = build_model_selection_report(records)
    assert report["selected_trained_reid_model"] == "higher_rank1"


def test_cross_domain_config_preserves_source_model_and_substitutes_target_test_data():
    source_cfg = {
        "experiment": {"output_dir": "source"},
        "model": {"name": "source_architecture"},
        "data": {
            "root": "source-root", "num_workers": 2, "pin_memory": False,
            "test": {"dataset": {"name": "market1501", "split": "test"}},
        },
    }
    target_cfg = {
        "data": {
            "root": "target-root", "num_workers": 4, "pin_memory": True,
            "test": {"dataset": {"name": "duke", "split": "test"}},
        },
    }
    cfg = build_cross_domain_config(source_cfg, target_cfg, "duke", "output")
    assert cfg["model"] == source_cfg["model"]
    assert cfg["data"]["test"] == target_cfg["data"]["test"]
    assert cfg["data"]["root"] == "target-root"
