import copy

import pytest
import torch

from reid.data.build import build_test_loader, build_train_loader
from reid.data.collate import test_collate_fn, train_collate_fn
from reid.data.protocol import (
    MARK_GALLERY,
    MARK_QUERY,
    ReIDEvalSample,
    ReIDTrainSample,
    validate_eval_dataset,
    validate_train_dataset,
)


class _TrainDataset:
    def __init__(self, labels=None, root=None, split=None, transform=None):
        del root, split, transform
        if labels is None:
            labels = [0, 0, 1, 1]
        self.labels = labels
        self.num_classes = len(set(labels)) if labels else 0

    def __len__(self):
        return len(self.labels)


class _EvalDataset:
    def __init__(self, marks=None, root=None, split=None, transform=None):
        del root, split, transform
        if marks is None:
            marks = [MARK_QUERY, MARK_GALLERY]
        self.pids = [1 for _ in marks]
        self.cams = [1 for _ in marks]
        self.marks = marks
        self.im_names = [f"{idx}.jpg" for idx, _ in enumerate(marks)]

    def __len__(self):
        return len(self.marks)


class _BadTrainDatasetLength:
    labels = [0]
    num_classes = 1

    def __len__(self):
        return 2


TRAIN_CFG = {
    "repro": {"seed": 42},
    "data": {
        "root": "/tmp/unused",
        "num_workers": 0,
        "pin_memory": False,
        "train": {
            "dataset": {"name": "market1501", "split": "trainval"},
            "images": {"size": [64, 32]},
            "aug": {},
            "batch": {"sampler": "random", "batch_size": 2},
        },
    },
    "loss": {
        "triplet": {"enabled": False},
        "id": {"enabled": True},
        "center": {"enabled": False},
    },
}


TEST_CFG = {
    "data": {
        "root": "/tmp/unused",
        "num_workers": 0,
        "pin_memory": False,
        "test": {
            "dataset": {"name": "market1501", "split": "test"},
            "images": {"size": [64, 32]},
            "aug": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "batch": {"size": 2},
            "loader": {"shuffle": False},
        },
    },
}


def test_train_sample_protocol_collates_to_training_batch():
    batch = [
        ReIDTrainSample(torch.randn(3, 64, 32), 0),
        ReIDTrainSample(torch.randn(3, 64, 32), 1),
    ]

    imgs, labels = train_collate_fn(batch)

    assert imgs.shape == (2, 3, 64, 32)
    assert labels.tolist() == [0, 1]
    assert labels.dtype == torch.long


def test_eval_sample_protocol_collates_to_eval_batch():
    batch = [
        ReIDEvalSample(torch.randn(3, 64, 32), 10, 2, "q.jpg", MARK_QUERY),
        ReIDEvalSample(torch.randn(3, 64, 32), 10, 3, "g.jpg", MARK_GALLERY),
    ]

    imgs, pids, camids, names, marks = test_collate_fn(batch)

    assert imgs.shape == (2, 3, 64, 32)
    assert pids.tolist() == [10, 10]
    assert camids.tolist() == [2, 3]
    assert names == ["q.jpg", "g.jpg"]
    assert marks.tolist() == [MARK_QUERY, MARK_GALLERY]


def test_eval_sample_uses_image_name_as_canonical_field():
    sample = ReIDEvalSample(
        image=torch.randn(3, 64, 32),
        pid=10,
        camid=2,
        image_name="query.jpg",
        mark=MARK_QUERY,
    )

    assert sample.image_name == "query.jpg"
    assert sample.name == "query.jpg"


def test_train_protocol_rejects_missing_metadata():
    class MissingMetadata:
        def __len__(self):
            return 1

    with pytest.raises(TypeError, match="labels and num_classes"):
        validate_train_dataset(MissingMetadata())


def test_train_protocol_rejects_label_length_mismatch():
    with pytest.raises(ValueError, match="labels length"):
        validate_train_dataset(_BadTrainDatasetLength())


def test_train_protocol_accepts_contiguous_non_negative_labels():
    validate_train_dataset(_TrainDataset(labels=[0, 0, 1, 1]))


def test_train_protocol_rejects_non_contiguous_labels():
    dataset = _TrainDataset(labels=[0, 0, 2, 2])
    dataset.num_classes = 3

    with pytest.raises(ValueError, match="contiguous non-negative"):
        validate_train_dataset(dataset)


def test_eval_protocol_requires_query_and_gallery_marks():
    with pytest.raises(ValueError, match="query"):
        validate_eval_dataset(_EvalDataset(marks=[MARK_GALLERY, MARK_GALLERY]))

    with pytest.raises(ValueError, match="gallery"):
        validate_eval_dataset(_EvalDataset(marks=[MARK_QUERY, MARK_QUERY]))


def test_eval_protocol_rejects_unknown_marks():
    with pytest.raises(ValueError, match="unsupported marks"):
        validate_eval_dataset(_EvalDataset(marks=[MARK_QUERY, 9, MARK_GALLERY]))


def test_build_train_loader_validates_dataset_protocol(monkeypatch):
    class MissingNumClassesDataset(_TrainDataset):
        def __init__(self, root=None, split=None, transform=None):
            super().__init__(root=root, split=split, transform=transform)
            del self.num_classes

    monkeypatch.setattr("reid.data.build.Market1501FromPartitions", MissingNumClassesDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)

    with pytest.raises(TypeError, match="labels and num_classes"):
        build_train_loader(copy.deepcopy(TRAIN_CFG))


def test_build_test_loader_validates_dataset_protocol(monkeypatch):
    class MissingMarksDataset(_EvalDataset):
        def __init__(self, root=None, split=None, transform=None):
            super().__init__(root=root, split=split, transform=transform)
            del self.marks

    monkeypatch.setattr("reid.data.build.Market1501TestFromPartitions", MissingMarksDataset)
    monkeypatch.setattr(
        "reid.data.build.build_test_tf",
        lambda image_size, mean, std: None,
    )

    with pytest.raises(TypeError, match="pids, cams, marks, and im_names"):
        build_test_loader(copy.deepcopy(TEST_CFG))


def test_build_train_loader_uses_raw_market1501_when_configured(monkeypatch):
    used = {}

    class RawTrainDataset(_TrainDataset):
        def __init__(self, root=None, split=None, transform=None):
            super().__init__(root=root, split=split, transform=transform)
            used["raw_train"] = True

    monkeypatch.setattr("reid.data.build.Market1501RawTrain", RawTrainDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(TRAIN_CFG)
    cfg["data"]["train"]["dataset"]["format"] = "raw"

    loader, _ = build_train_loader(cfg)

    assert used["raw_train"] is True
    assert loader.dataset.num_classes == 2


def test_build_test_loader_uses_raw_market1501_when_configured(monkeypatch):
    used = {}

    class RawTestDataset(_EvalDataset):
        def __init__(self, root=None, split=None, transform=None):
            super().__init__(root=root, split=split, transform=transform)
            used["raw_test"] = True

    monkeypatch.setattr("reid.data.build.Market1501RawTest", RawTestDataset)
    monkeypatch.setattr(
        "reid.data.build.build_test_tf",
        lambda image_size, mean, std: None,
    )
    cfg = copy.deepcopy(TEST_CFG)
    cfg["data"]["test"]["dataset"]["format"] = "raw"

    loader = build_test_loader(cfg)

    assert used["raw_test"] is True
    assert loader.dataset.marks == [MARK_QUERY, MARK_GALLERY]


def test_build_loaders_reject_unknown_market1501_format(monkeypatch):
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    train_cfg = copy.deepcopy(TRAIN_CFG)
    train_cfg["data"]["train"]["dataset"]["format"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported Market1501 train dataset format"):
        build_train_loader(train_cfg)

    monkeypatch.setattr(
        "reid.data.build.build_test_tf",
        lambda image_size, mean, std: None,
    )
    test_cfg = copy.deepcopy(TEST_CFG)
    test_cfg["data"]["test"]["dataset"]["format"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported Market1501 test dataset format"):
        build_test_loader(test_cfg)


def test_build_train_loader_uses_processed_cuhk03_when_configured(monkeypatch, capsys):
    used = {}

    class CUHKTrainDataset(_TrainDataset):
        def __init__(
            self,
            root=None,
            split=None,
            image_type="detected",
            protocol="new",
            split_id=0,
            transform=None,
            ):
                super().__init__(root=root, split=split, transform=transform)
                self.split = split
                self.image_type = image_type
                self.protocol = protocol
                self.split_id = split_id
                used.update({
                "root": root,
                "split": split,
                "image_type": image_type,
                "protocol": protocol,
                "split_id": split_id,
            })

    monkeypatch.setattr("reid.data.build.CUHK03ProcessedTrain", CUHKTrainDataset)
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    cfg = copy.deepcopy(TRAIN_CFG)
    cfg["data"]["train"]["dataset"] = {
        "name": "cuhk03",
        "split": "trainval",
        "format": "processed",
        "image_type": "labeled",
        "protocol": "classic",
        "split_id": 2,
    }

    loader, _ = build_train_loader(cfg)
    out = capsys.readouterr().out

    assert loader.dataset.num_classes == 2
    assert "[CUHK03] format=processed image_type=labeled split=trainval" in out
    assert "num images: 4" in out
    assert "num identities: 2" in out
    assert "num query images: n/a" in out
    assert "num gallery images: n/a" in out
    assert used == {
        "root": "/tmp/unused",
        "split": "trainval",
        "image_type": "labeled",
        "protocol": "classic",
        "split_id": 2,
    }


def test_build_test_loader_uses_processed_cuhk03_when_configured(monkeypatch, capsys):
    used = {}

    class CUHKTestDataset(_EvalDataset):
        def __init__(
            self,
            root=None,
            split=None,
            image_type="detected",
            protocol="new",
            split_id=0,
            transform=None,
            ):
                super().__init__(root=root, split=split, transform=transform)
                self.split = split
                self.image_type = image_type
                self.protocol = protocol
                self.split_id = split_id
                used.update({
                "root": root,
                "split": split,
                "image_type": image_type,
                "protocol": protocol,
                "split_id": split_id,
            })

    monkeypatch.setattr("reid.data.build.CUHK03ProcessedTest", CUHKTestDataset)
    monkeypatch.setattr(
        "reid.data.build.build_test_tf",
        lambda image_size, mean, std: None,
    )
    cfg = copy.deepcopy(TEST_CFG)
    cfg["data"]["test"]["dataset"] = {
        "name": "cuhk03",
        "split": "test",
        "format": "processed",
        "image_type": "detected",
        "protocol": "new",
        "split_id": 0,
    }

    loader = build_test_loader(cfg)
    out = capsys.readouterr().out

    assert loader.dataset.marks == [MARK_QUERY, MARK_GALLERY]
    assert "[CUHK03] format=processed image_type=detected split=test" in out
    assert "num images: 2" in out
    assert "num identities: n/a" in out
    assert "num query images: 1" in out
    assert "num gallery images: 1" in out
    assert used == {
        "root": "/tmp/unused",
        "split": "test",
        "image_type": "detected",
        "protocol": "new",
        "split_id": 0,
    }


def test_build_loaders_reject_unknown_cuhk03_format(monkeypatch):
    monkeypatch.setattr("reid.data.build.build_train_tf", lambda image_size, aug_cfg: None)
    train_cfg = copy.deepcopy(TRAIN_CFG)
    train_cfg["data"]["train"]["dataset"] = {
        "name": "cuhk03",
        "split": "trainval",
        "format": "raw",
    }

    with pytest.raises(ValueError, match="Unsupported CUHK03 train dataset format"):
        build_train_loader(train_cfg)

    monkeypatch.setattr(
        "reid.data.build.build_test_tf",
        lambda image_size, mean, std: None,
    )
    test_cfg = copy.deepcopy(TEST_CFG)
    test_cfg["data"]["test"]["dataset"] = {
        "name": "cuhk03",
        "split": "test",
        "format": "raw",
    }

    with pytest.raises(ValueError, match="Unsupported CUHK03 test dataset format"):
        build_test_loader(test_cfg)
