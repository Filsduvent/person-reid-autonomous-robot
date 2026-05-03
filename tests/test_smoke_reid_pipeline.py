import argparse

import pytest
import torch

from scripts.smoke_reid_pipeline import (
    _base_overrides,
    describe_eval_batch,
    describe_train_batch,
)


def test_smoke_base_overrides_disable_pretrained_and_num_workers_by_default():
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
        "model.backbone.pretrained=false",
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
