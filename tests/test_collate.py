import torch

from reid.data.collate import test_collate_fn, train_collate_fn


def test_train_collate_fn_returns_images_and_long_labels():
    batch = [
        (torch.randn(3, 64, 32), 1),
        (torch.randn(3, 64, 32), 2),
    ]

    imgs, labels = train_collate_fn(batch)

    assert imgs.ndim == 4
    assert imgs.shape[0] == 2
    assert labels.dtype == torch.long


def test_test_collate_fn_returns_expected_metadata_types():
    batch = [
        (torch.randn(3, 64, 32), 1, 2, "a.jpg", 0),
        (torch.randn(3, 64, 32), 3, 4, "b.jpg", 1),
    ]

    imgs, pids, camids, names, marks = test_collate_fn(batch)

    assert imgs.ndim == 4
    assert pids.dtype == torch.long
    assert camids.dtype == torch.long
    assert isinstance(names, list)
    assert names == ["a.jpg", "b.jpg"]
    assert marks.dtype == torch.long
