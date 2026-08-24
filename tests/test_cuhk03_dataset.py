import pickle

import pytest
from PIL import Image

from reid.data.cuhk03 import (
    CUHK03ProcessedTest,
    CUHK03ProcessedTrain,
    parse_processed_name,
    parse_processed_reid_name,
)


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 8), color=(0, 255, 0)).save(path)


def _pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)


def test_parse_processed_cuhk03_name_returns_pid_and_camid():
    pid, camid = parse_processed_reid_name("00000012_0001_00000000.jpg")

    assert pid == 12
    assert camid == 1


def test_parse_processed_cuhk03_name_supports_png_extension():
    pid, camid = parse_processed_reid_name("00000012_0001_00000000.png")

    assert pid == 12
    assert camid == 1


def test_parse_processed_name_alias_keeps_existing_import_compatibility():
    assert parse_processed_name("00000012_0001_00000000.JPG") == (12, 1)


@pytest.mark.parametrize(
    "name",
    [
        "00000012_c001_00000000.jpg",
        "000012_0001_00000000.jpg",
        "00000012_001_00000000.jpg",
        "00000012_0001_0000000.jpg",
        "00000012_0001_00000000.bmp",
    ],
)
def test_parse_processed_reid_name_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="8digits_4digits_8digits"):
        parse_processed_reid_name(name)


def test_cuhk03_processed_train_uses_selected_image_type_and_returns_image_label(tmp_path):
    images_dir = tmp_path / "cuhk03" / "detected" / "images"
    _image(images_dir / "00000002_0001_00000000.jpg")
    _image(images_dir / "00000007_0002_00000000.jpg")
    _pickle(
        tmp_path / "cuhk03" / "detected" / "partitions.pkl",
        {
            "trainval_im_names": [
                "00000002_0001_00000000.jpg",
                "00000007_0002_00000000.jpg",
            ],
            "trainval_ids2labels": {2: 0, 7: 1},
        },
    )

    dataset = CUHK03ProcessedTrain(
        root=str(tmp_path),
        split="trainval",
        image_type="detected",
        protocol="processed_partition",
        transform=lambda img: img,
    )

    assert dataset.image_type == "detected"
    assert dataset.protocol == "processed_partition"
    assert dataset.split_id is None
    assert dataset.labels == [0, 1]
    assert dataset.pids == [2, 7]
    assert dataset.cams == [1, 2]
    assert dataset.num_classes == 2

    img, label = dataset[1]

    assert img.size == (4, 8)
    assert label == 1


def test_cuhk03_processed_train_supports_train_split_with_matching_label_key(tmp_path):
    images_dir = tmp_path / "cuhk03" / "detected" / "images"
    _image(images_dir / "00000003_0001_00000000.jpg")
    _pickle(
        tmp_path / "cuhk03" / "detected" / "partitions.pkl",
        {
            "train_im_names": ["00000003_0001_00000000.jpg"],
            "train_ids2labels": {3: 0},
        },
    )

    dataset = CUHK03ProcessedTrain(root=str(tmp_path), split="train", image_type="detected")

    assert dataset.im_names == ["00000003_0001_00000000.jpg"]
    assert dataset.im_dir == str(images_dir)
    assert dataset.labels == [0]
    assert dataset.num_classes == 1


def test_cuhk03_processed_train_requires_train_split_and_label_key(tmp_path):
    images_dir = tmp_path / "cuhk03" / "detected" / "images"
    _image(images_dir / "00000003_0001_00000000.jpg")
    _pickle(
        tmp_path / "cuhk03" / "detected" / "partitions.pkl",
        {
            "trainval_im_names": ["00000003_0001_00000000.jpg"],
        },
    )

    with pytest.raises(ValueError, match="CUHK03 processed train supports"):
        CUHK03ProcessedTrain(root=str(tmp_path), split="test", image_type="detected")

    with pytest.raises(KeyError, match="trainval_ids2labels"):
        CUHK03ProcessedTrain(root=str(tmp_path), split="trainval", image_type="detected")


def test_cuhk03_processed_test_uses_selected_image_type_and_returns_eval_tuple(tmp_path):
    images_dir = tmp_path / "cuhk03" / "labeled" / "images"
    _image(images_dir / "00000002_0001_00000000.jpg")
    _image(images_dir / "00000002_0002_00000000.jpg")
    _pickle(
        tmp_path / "cuhk03" / "labeled" / "partitions.pkl",
        {
            "test_im_names": [
                "00000002_0001_00000000.jpg",
                "00000002_0002_00000000.jpg",
            ],
            "test_marks": [0, 1],
        },
    )

    dataset = CUHK03ProcessedTest(
        root=str(tmp_path),
        split="test",
        image_type="labeled",
        protocol="processed_partition",
        transform=lambda img: img,
    )

    assert dataset.image_type == "labeled"
    assert dataset.protocol == "processed_partition"
    assert dataset.split_id is None
    assert dataset.pids.tolist() == [2, 2]
    assert dataset.cams.tolist() == [1, 2]
    assert dataset.marks.tolist() == [0, 1]

    img, pid, camid, image_name, mark = dataset[0]

    assert img.size == (4, 8)
    assert pid == 2
    assert camid == 1
    assert image_name == "00000002_0001_00000000.jpg"
    assert mark == 0


def test_cuhk03_processed_test_supports_val_split_and_rejects_train_split(tmp_path):
    images_dir = tmp_path / "cuhk03" / "detected" / "images"
    _image(images_dir / "00000004_0002_00000000.jpg")
    _image(images_dir / "00000004_0003_00000000.jpg")
    _pickle(
        tmp_path / "cuhk03" / "detected" / "partitions.pkl",
        {
            "val_im_names": [
                "00000004_0002_00000000.jpg",
                "00000004_0003_00000000.jpg",
            ],
            "val_marks": [0, 1],
        },
    )

    dataset = CUHK03ProcessedTest(root=str(tmp_path), split="val", image_type="detected")

    assert dataset.pids.tolist() == [4, 4]
    assert dataset.cams.tolist() == [2, 3]
    assert dataset.marks.tolist() == [0, 1]

    with pytest.raises(ValueError, match="CUHK03 processed test supports"):
        CUHK03ProcessedTest(root=str(tmp_path), split="train", image_type="detected")


def test_cuhk03_processed_rejects_invalid_image_type(tmp_path):
    with pytest.raises(ValueError, match="Unsupported CUHK03 image_type"):
        CUHK03ProcessedTrain(root=str(tmp_path), split="trainval", image_type="raw")


@pytest.mark.parametrize("protocol", ["new", "classic"])
def test_cuhk03_processed_rejects_protocols_not_encoded_by_partition_files(tmp_path, protocol):
    with pytest.raises(ValueError, match="do not encode"):
        CUHK03ProcessedTrain(
            root=str(tmp_path), split="trainval", image_type="detected", protocol=protocol
        )


def test_cuhk03_processed_rejects_unrepresented_split_id(tmp_path):
    with pytest.raises(ValueError, match="split_id"):
        CUHK03ProcessedTrain(
            root=str(tmp_path), split="trainval", image_type="detected", split_id=0
        )


def test_cuhk03_processed_train_requires_images_and_partitions(tmp_path):
    images_dir = tmp_path / "cuhk03" / "detected" / "images"
    part_file = tmp_path / "cuhk03" / "detected" / "partitions.pkl"

    with pytest.raises(FileNotFoundError, match=str(images_dir)):
        CUHK03ProcessedTrain(root=str(tmp_path), split="trainval", image_type="detected")

    images_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=str(part_file)):
        CUHK03ProcessedTrain(root=str(tmp_path), split="trainval", image_type="detected")


def test_cuhk03_processed_test_requires_images_and_partitions(tmp_path):
    images_dir = tmp_path / "cuhk03" / "labeled" / "images"
    part_file = tmp_path / "cuhk03" / "labeled" / "partitions.pkl"

    with pytest.raises(FileNotFoundError, match=str(images_dir)):
        CUHK03ProcessedTest(root=str(tmp_path), split="test", image_type="labeled")

    images_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=str(part_file)):
        CUHK03ProcessedTest(root=str(tmp_path), split="test", image_type="labeled")


def test_cuhk03_processed_test_requires_expected_partition_keys(tmp_path):
    images_dir = tmp_path / "cuhk03" / "detected" / "images"
    _image(images_dir / "00000004_0002_00000000.jpg")
    _pickle(
        tmp_path / "cuhk03" / "detected" / "partitions.pkl",
        {
            "test_im_names": ["00000004_0002_00000000.jpg"],
        },
    )

    with pytest.raises(KeyError, match="test_marks"):
        CUHK03ProcessedTest(root=str(tmp_path), split="test", image_type="detected")
