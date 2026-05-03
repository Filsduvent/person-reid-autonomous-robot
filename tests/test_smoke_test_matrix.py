import argparse
import sys

from scripts.smoke_test import DATASET_CONFIGS, build_smoke_command


def _args(**kwargs):
    defaults = {
        "root": "/datasets",
        "device": "cpu",
        "opts": [],
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_smoke_matrix_covers_all_locked_datasets():
    assert set(DATASET_CONFIGS) == {"market1501", "duke", "cuhk03", "msmt17"}


def test_smoke_command_runs_train_with_one_epoch_eval_checkpoint_overrides():
    cmd = build_smoke_command("market1501", _args())

    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/train.py")
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1].endswith("configs/baseline_market1501_resnet50_triplet.yaml")
    assert "-o" in cmd

    overrides = cmd[cmd.index("-o") + 1:]
    assert "data.root=/datasets" in overrides
    assert "system.device=cpu" in overrides
    assert "model.backbone.pretrained=false" in overrides
    assert "train.epochs=1" in overrides
    assert "train.eval_interval=1" in overrides
    assert "data.train.batch.P=4" in overrides
    assert "data.train.batch.K=2" in overrides
    assert "data.train.batch.batch_size=8" in overrides
    assert "data.test.batch.size=8" in overrides
    assert "experiment.name=smoke_market1501_resnet50" in overrides
    assert "experiment.output_dir=exp/smoke_market1501_resnet50" in overrides


def test_smoke_command_allows_extra_overrides_to_win_by_order():
    cmd = build_smoke_command(
        "msmt17",
        _args(device="cuda", root="", opts=["data.test.batch.size=4", "logging.tensorboard=true"]),
    )
    overrides = cmd[cmd.index("-o") + 1:]

    assert "system.device=cuda" in overrides
    assert "data.root=/datasets" not in overrides
    assert overrides.index("data.test.batch.size=8") < overrides.index("data.test.batch.size=4")
    assert overrides.index("logging.tensorboard=false") < overrides.index("logging.tensorboard=true")
