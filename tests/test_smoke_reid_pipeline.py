import argparse

import pytest
import torch

from scripts.smoke_reid_pipeline import (
    _base_overrides,
    describe_eval_batch,
    describe_train_batch,
)


def test_smoke_base_overrides_do_not_inject_architecture_fields():
    args = argparse.Namespace(
        device="cpu",
        root="/data/root",
        use_config_pretrained=False,
        opts=["data.test.batch.size=8"],
    )

    overrides = _base_overrides(args)

    assert overrides == [
        "data.num_workers=0",
        "system.device=cpu",
        "data.root=/data/root",
        "data.test.batch.size=8",
    ]


def test_smoke_base_overrides_can_honor_config_pretrained():
    args = argparse.Namespace(
        device="cuda",
        root="",
        use_config_pretrained=True,
        opts=[],
    )

    overrides = _base_overrides(args)

    assert overrides == [
        "data.num_workers=0",
        "system.device=cuda",
    ]


def test_describe_train_batch_validates_contract():
    info = describe_train_batch((torch.randn(2, 3, 64, 32), torch.tensor([0, 1])))

    assert info == {
        "images": (2, 3, 64, 32),
        "labels": (2,),
        "dtype": "torch.int64",
    }


def test_describe_train_batch_rejects_bad_shape():
    with pytest.raises(ValueError, match="NCHW"):
        describe_train_batch((torch.randn(3, 64, 32), torch.tensor([0])))

    with pytest.raises(ValueError, match="match label count"):
        describe_train_batch((torch.randn(2, 3, 64, 32), torch.tensor([0])))


def test_describe_eval_batch_validates_contract():
    info = describe_eval_batch(
        (
            torch.randn(2, 3, 64, 32),
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            ["q.jpg", "g.jpg"],
            torch.tensor([0, 1]),
        )
    )

    assert info == {
        "images": (2, 3, 64, 32),
        "pids": (2,),
        "camids": (2,),
        "marks": (2,),
        "first_name": "q.jpg",
    }


def test_describe_eval_batch_rejects_bad_metadata_lengths():
    with pytest.raises(ValueError, match="pids count"):
        describe_eval_batch(
            (
                torch.randn(2, 3, 64, 32),
                torch.tensor([0]),
                torch.tensor([0, 2]),
                ["q.jpg", "g.jpg"],
                torch.tensor([0, 1]),
            )
        )

    with pytest.raises(ValueError, match="names count"):
        describe_eval_batch(
            (
                torch.randn(2, 3, 64, 32),
                torch.tensor([0, 1]),
                torch.tensor([0, 2]),
                ["q.jpg"],
                torch.tensor([0, 1]),
            )
        )


@pytest.mark.parametrize("model_name", ["pcb", "reid_baseline"])
@pytest.mark.parametrize("use_pretrained", [False, True])
def test_smoke_passes_generic_initialization_control(monkeypatch, tmp_path, model_name, use_pretrained):
    from types import SimpleNamespace
    from scripts import smoke_reid_pipeline as smoke

    cfg = {
        "model": {"name": model_name},
        "experiment": {"output_dir": str(tmp_path)},
        "system": {"device": "cpu"},
    }
    args = argparse.Namespace(config="synthetic", device="cpu", root="", opts=[],
                              use_config_pretrained=use_pretrained,
                              skip_batch=False, skip_model=False)
    train_batch = (torch.randn(2, 3, 64, 32), torch.tensor([0, 1]))
    eval_batch = (train_batch[0], train_batch[1], torch.tensor([0, 1]),
                  ["q.jpg", "g.jpg"], torch.tensor([0, 1]))

    class Loader(list):
        dataset = SimpleNamespace()

    monkeypatch.setattr(smoke, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(smoke, "validate_config", lambda *a, **kw: None)
    monkeypatch.setattr(smoke, "validate_reid_config", lambda *a, **kw: None)
    monkeypatch.setattr(smoke, "save_run_artifacts", lambda *a, **kw: {"command": "x", "environment": "y"})
    monkeypatch.setattr(smoke, "build_train_loader", lambda cfg: (Loader([train_batch]), 3))
    monkeypatch.setattr(smoke, "build_test_loader", lambda cfg: Loader([eval_batch]))
    monkeypatch.setattr(smoke, "select_device", lambda *a: (torch.device("cpu"), None))

    class ReachedBuilder(Exception):
        pass

    def capture_builder(actual_cfg, num_classes, *, initialize_pretrained):
        assert actual_cfg["model"] == {"name": model_name}
        assert num_classes == 3
        assert initialize_pretrained is use_pretrained
        raise ReachedBuilder

    monkeypatch.setattr(smoke, "build_model", capture_builder)
    with pytest.raises(ReachedBuilder):
        smoke.run_smoke(args)
