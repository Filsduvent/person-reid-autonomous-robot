import copy

import pytest
import torch

from reid.data.build import build_train_loader
from reid.data.samplers import PKBatchSampler


def test_pk_batch_sampler_yields_expected_batch_structure():
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    sampler = PKBatchSampler(labels, P=2, K=2, seed=42)

    batch = next(iter(sampler))

    assert len(batch) == 4
    batch_labels = [labels[i] for i in batch]
    assert len(set(batch_labels)) == 2
    for lab in set(batch_labels):
        assert batch_labels.count(lab) == 2


def test_pk_batch_sampler_len_is_finite_and_positive():
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    sampler = PKBatchSampler(labels, P=2, K=2, seed=42)

    assert len(sampler) > 0
    assert len(sampler) < 10**9


def test_pk_batch_sampler_iteration_is_finite():
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    sampler = PKBatchSampler(labels, P=2, K=2, seed=42)

    batches = list(iter(sampler))

    assert len(batches) == len(sampler)
    for batch in batches:
        assert len(batch) == 4


BASE_CFG = {
    "repro": {"seed": 42},
    "data": {
        "root": "/tmp/unused",
        "num_workers": 0,
        "pin_memory": False,
        "train": {
            "dataset": {"name": "market1501", "split": "trainval"},
            "images": {"size": [256, 128]},
            "aug": {},
            "batch": {
                "sampler": "pk",
                "P": 2,
                "K": 2,
                "batch_size": 4,
            },
        },
    },
    "loss": {
        "triplet": {"enabled": True},
        "id": {"enabled": True},
        "center": {"enabled": False},
    },
}


class _DummyDataset:
    def __init__(self, root, split, transform):
        del root, split, transform
        self.labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.num_classes = 4

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.randn(3, 256, 128), self.labels[idx]


def test_build_train_loader_pk_mode_has_finite_length(monkeypatch):
    monkeypatch.setattr("reid.data.build.Market1501FromPartitions", _DummyDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(BASE_CFG)

    loader, batch_size = build_train_loader(cfg)

    assert batch_size == 4
    assert len(loader) > 0
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape[0] == 4
    assert batch_labels.shape[0] == 4


def test_build_train_loader_random_mode_works_for_id_only(monkeypatch):
    monkeypatch.setattr("reid.data.build.Market1501FromPartitions", _DummyDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(BASE_CFG)
    cfg["loss"]["triplet"]["enabled"] = False
    cfg["data"]["train"]["batch"]["sampler"] = "random"
    cfg["data"]["train"]["batch"]["batch_size"] = 4

    loader, batch_size = build_train_loader(cfg)

    assert batch_size == 4
    assert len(loader) == len(_DummyDataset(None, None, None)) // 4
    batch_imgs, batch_labels = next(iter(loader))
    assert batch_imgs.shape[0] == 4
    assert batch_labels.shape[0] == 4


def test_build_train_loader_rejects_random_sampler_with_triplet(monkeypatch):
    monkeypatch.setattr("reid.data.build.Market1501FromPartitions", _DummyDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(BASE_CFG)
    cfg["data"]["train"]["batch"]["sampler"] = "random"

    with pytest.raises(
        ValueError,
        match="Triplet loss requires sampler='pk' so each batch has positive pairs.",
    ):
        build_train_loader(cfg)


def test_build_train_loader_rejects_mismatched_pk_batch_size(monkeypatch):
    monkeypatch.setattr("reid.data.build.Market1501FromPartitions", _DummyDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(BASE_CFG)
    cfg["data"]["train"]["batch"]["batch_size"] = 5

    with pytest.raises(ValueError, match=r"PK sampler requires batch_size == P\*K"):
        build_train_loader(cfg)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("P", 1, "PK sampler requires P > 1"),
        ("K", 1, "PK sampler requires K > 1"),
    ],
)
def test_build_train_loader_rejects_invalid_pk_shape(monkeypatch, field, value, expected_message):
    monkeypatch.setattr("reid.data.build.Market1501FromPartitions", _DummyDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(BASE_CFG)
    cfg["data"]["train"]["batch"][field] = value
    cfg["data"]["train"]["batch"]["batch_size"] = cfg["data"]["train"]["batch"]["P"] * cfg["data"]["train"]["batch"]["K"]

    with pytest.raises(ValueError, match=expected_message):
        build_train_loader(cfg)
