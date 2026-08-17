import json

from reid.utils.metrics_artifacts import (
    build_metric_payload,
    save_final_eval_metrics,
    save_standalone_eval_metrics,
    save_train_eval_metrics,
)
from scripts.train import create_experiment_dirs


CFG = {
    "model": {
        "name": "reid_baseline",
        "backbone": {
            "name": "resnet50",
        },
    },
    "data": {
        "test": {
            "dataset": {
                "name": "market1501",
                "split": "test",
            }
        }
    },
}


SCORES = {
    "mAP": 0.1,
    "mINP": 0.2,
    "Rank1": 0.3,
    "Rank5": 0.4,
    "Rank10": 0.5,
}


RERANK_SCORES = {
    **SCORES,
    "rerank_mAP": 0.6,
    "rerank_mINP": 0.7,
    "rerank_Rank1": 0.8,
    "rerank_Rank5": 0.9,
    "rerank_Rank10": 1.0,
}


def test_experiment_dirs_include_locked_artifact_layout(tmp_path):
    create_experiment_dirs(str(tmp_path))

    assert (tmp_path / "metrics").is_dir()
    assert (tmp_path / "checkpoints").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "plots").is_dir()
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "tensorboard").is_dir()
    assert (tmp_path / "tb").exists()


def test_metric_payload_uses_locked_schema_and_model_label():
    payload = build_metric_payload(CFG, SCORES, epoch=120, checkpoint_name="ckpt_best.pth")

    assert payload == {
        "dataset": "market1501",
        "split": "test",
        "model": "resnet50_baseline",
        "epoch": 120,
        "checkpoint": "ckpt_best.pth",
        "mAP": 0.1,
        "mINP": 0.2,
        "Rank1": 0.3,
        "Rank5": 0.4,
        "Rank10": 0.5,
    }


def test_metric_payload_preserves_rerank_metrics_separately():
    payload = build_metric_payload(CFG, RERANK_SCORES, epoch=120, checkpoint_name="ckpt_best.pth")

    assert payload["mAP"] == 0.1
    assert payload["Rank1"] == 0.3
    assert payload["rerank_mAP"] == 0.6
    assert payload["rerank_Rank1"] == 0.8


def test_train_metric_files_share_same_schema(tmp_path):
    paths = save_train_eval_metrics(str(tmp_path), CFG, epoch=3, scores=RERANK_SCORES, checkpoint_name="ckpt_last.pth")

    latest = json.loads((tmp_path / "latest_val.json").read_text(encoding="utf-8"))
    epoch_payload = json.loads((tmp_path / "val_epoch_003.json").read_text(encoding="utf-8"))

    assert latest == epoch_payload == paths["payload"]
    assert latest["model"] == "resnet50_baseline"
    assert latest["rerank_Rank10"] == 1.0


def test_final_metric_file_uses_same_schema(tmp_path):
    paths = save_final_eval_metrics(str(tmp_path), CFG, epoch=4, scores=SCORES, checkpoint_name="ckpt_best.pth")

    final_payload = json.loads((tmp_path / "final_test.json").read_text(encoding="utf-8"))

    assert paths["final"].endswith("final_test.json")
    assert final_payload == paths["payload"]
    assert final_payload["checkpoint"] == "ckpt_best.pth"


def test_standalone_eval_writes_eval_and_final_metric_files(tmp_path):
    paths = save_standalone_eval_metrics(str(tmp_path), CFG, "ckpt_best.pth", SCORES, epoch=9)

    eval_payload = json.loads((tmp_path / "eval_ckpt_best.json").read_text(encoding="utf-8"))
    final_payload = json.loads((tmp_path / "final_test.json").read_text(encoding="utf-8"))

    assert paths["eval"].endswith("eval_ckpt_best.json")
    assert paths["final"].endswith("final_test.json")
    assert eval_payload == final_payload == paths["payload"]
