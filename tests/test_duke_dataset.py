import pickle

import pytest
from PIL import Image

from reid.data.duke import (
    DukeProcessedTest,
    DukeProcessedTrain,
    DukeRawTest,
    DukeRawTrain,
    parse_duke_raw_name,
    parse_processed_reid_name,
    parse_raw_duke_dir,
)


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 8), color=(0, 0, 255)).save(path)


def _pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)


def test_parse_duke_name_returns_pid_and_zero_based_camid():
    pid, camid = parse_duke_raw_name("0002_c1_f0046185.jpg")

    assert pid == 2
    assert camid == 0


def test_parse_duke_name_rejects_out_of_range_camid():
    with pytest.raises(AssertionError, match="Invalid Duke camid"):
        parse_duke_raw_name("0002_c9_f0046182.jpg")


def test_parse_processed_reid_name_for_duke_processed_filenames():
    pid, camid = parse_processed_reid_name("00000002_0001_00000000.jpg")

    assert pid == 2
    assert camid == 1


def test_parse_raw_duke_dir_relabels_train_pids_and_zero_bases_camids(tmp_path):
    raw_dir = tmp_path / "bounding_box_train"
    _image(raw_dir / "0002_c1_f0046182.jpg")
    _image(raw_dir / "0007_c8_f0046183.jpg")
    _image(raw_dir / "0002_c3_f0046184.jpg")
    _image(raw_dir / "-1_c1_f0046185.jpg")

    records = parse_raw_duke_dir(str(raw_dir), relabel=True)

    assert records == [
        (str(raw_dir / "0002_c1_f0046182.jpg"), 0, 0),
        (str(raw_dir / "0002_c3_f0046184.jpg"), 0, 2),
        (str(raw_dir / "0007_c8_f0046183.jpg"), 1, 7),
    ]


def test_parse_raw_duke_dir_keeps_eval_pids_unrelabeled(tmp_path):
    raw_dir = tmp_path / "query"
    _image(raw_dir / "0002_c1_f0046182.jpg")
    _image(raw_dir / "0007_c8_f0046183.jpg")

    records = parse_raw_duke_dir(str(raw_dir), relabel=False)

    assert records == [
        (str(raw_dir / "0002_c1_f0046182.jpg"), 2, 0),
        (str(raw_dir / "0007_c8_f0046183.jpg"), 7, 7),
    ]


def test_duke_raw_train_exposes_samples_labels_and_num_classes(tmp_path):
    train_dir = tmp_path / "duke" / "DukeMTMC-reID" / "bounding_box_train"
    _image(train_dir / "0002_c1_f0046182.jpg")
    _image(train_dir / "0007_c8_f0046183.jpg")
    _image(train_dir / "0002_c3_f0046184.jpg")
    _image(train_dir / "-1_c1_f0046185.jpg")

    dataset = DukeRawTrain(root=str(tmp_path), transform=lambda img: img)

    assert dataset.samples == [
        (str(train_dir / "0002_c1_f0046182.jpg"), 2, 0, 0),
        (str(train_dir / "0002_c3_f0046184.jpg"), 2, 2, 0),
        (str(train_dir / "0007_c8_f0046183.jpg"), 7, 7, 1),
    ]
    assert dataset.labels == [0, 0, 1]
    assert dataset.num_classes == 2

    img, label = dataset[0]

    assert img.size == (4, 8)
    assert label == 0


def test_duke_raw_test_keeps_query_first_then_gallery(tmp_path):
    raw_root = tmp_path / "duke" / "DukeMTMC-reID"
    query_dir = raw_root / "query"
    gallery_dir = raw_root / "bounding_box_test"
    _image(query_dir / "0002_c1_f0046182.jpg")
    _image(query_dir / "0007_c8_f0046183.jpg")
    _image(gallery_dir / "0002_c2_f0046184.jpg")
    _image(gallery_dir / "0008_c5_f0046185.jpg")
    _image(gallery_dir / "-1_c1_f0046186.jpg")

    dataset = DukeRawTest(root=str(tmp_path), transform=lambda img: img)

    assert dataset.samples == [
        (str(query_dir / "0002_c1_f0046182.jpg"), 2, 0, 0),
        (str(query_dir / "0007_c8_f0046183.jpg"), 7, 7, 0),
        (str(gallery_dir / "0002_c2_f0046184.jpg"), 2, 1, 1),
        (str(gallery_dir / "0008_c5_f0046185.jpg"), 8, 4, 1),
    ]
    assert dataset.pids.tolist() == [2, 7, 2, 8]
    assert dataset.cams.tolist() == [0, 7, 1, 4]
    assert dataset.marks.tolist() == [0, 0, 1, 1]
    assert dataset.num_query == 2
    assert dataset.num_gallery == 2

    img, pid, camid, image_name, mark = dataset[2]

    assert img.size == (4, 8)
    assert pid == 2
    assert camid == 1
    assert image_name == "0002_c2_f0046184.jpg"
    assert mark == 1


def test_duke_raw_requires_expected_directories(tmp_path):
    train_dir = tmp_path / "duke" / "DukeMTMC-reID" / "bounding_box_train"
    query_dir = tmp_path / "duke" / "DukeMTMC-reID" / "query"
    gallery_dir = tmp_path / "duke" / "DukeMTMC-reID" / "bounding_box_test"

    with pytest.raises(FileNotFoundError, match=str(train_dir)):
        DukeRawTrain(root=str(tmp_path))

    gallery_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=str(query_dir)):
        DukeRawTest(root=str(tmp_path))

    query_dir.mkdir(parents=True)
    gallery_dir.rmdir()
    with pytest.raises(FileNotFoundError, match=str(gallery_dir)):
        DukeRawTest(root=str(tmp_path))


def test_duke_processed_train_returns_image_label_and_num_classes(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    _image(images_dir / "00000002_0001_00000000.jpg")
    _image(images_dir / "00000007_0002_00000000.jpg")
    _pickle(
        tmp_path / "duke" / "partitions.pkl",
        {
            "trainval_im_names": [
                "00000002_0001_00000000.jpg",
                "00000007_0002_00000000.jpg",
            ],
            "trainval_ids2labels": {2: 0, 7: 1},
        },
    )

    dataset = DukeProcessedTrain(root=str(tmp_path), split="trainval", transform=lambda img: img)

    assert dataset.im_dir == str(images_dir)
    assert dataset.im_names == [
        "00000002_0001_00000000.jpg",
        "00000007_0002_00000000.jpg",
    ]
    assert dataset.pids == [2, 7]
    assert dataset.cams == [1, 2]
    assert dataset.labels == [0, 1]
    assert dataset.num_classes == 2

    img, label = dataset[1]

    assert img.size == (4, 8)
    assert label == 1


def test_duke_processed_train_supports_train_split(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    _image(images_dir / "00000003_0001_00000000.jpg")
    _pickle(
        tmp_path / "duke" / "partitions.pkl",
        {
            "train_im_names": ["00000003_0001_00000000.jpg"],
            "train_ids2labels": {3: 0},
        },
    )

    dataset = DukeProcessedTrain(root=str(tmp_path), split="train")

    assert dataset.labels == [0]
    assert dataset.num_classes == 1


def test_duke_processed_test_returns_eval_tuple(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    _image(images_dir / "00000002_0001_00000000.jpg")
    _image(images_dir / "00000002_0002_00000000.jpg")
    _pickle(
        tmp_path / "duke" / "partitions.pkl",
        {
            "test_im_names": [
                "00000002_0001_00000000.jpg",
                "00000002_0002_00000000.jpg",
            ],
            "test_marks": [0, 1],
        },
    )

    dataset = DukeProcessedTest(root=str(tmp_path), split="test", transform=lambda img: img)

    assert dataset.pids.tolist() == [2, 2]
    assert dataset.cams.tolist() == [1, 2]
    assert dataset.marks.tolist() == [0, 1]
    assert dataset.num_query == 1
    assert dataset.num_gallery == 1

    img, pid, camid, image_name, mark = dataset[0]

    assert img.size == (4, 8)
    assert pid == 2
    assert camid == 1
    assert image_name == "00000002_0001_00000000.jpg"
    assert mark == 0


def test_duke_processed_test_supports_val_split(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    _image(images_dir / "00000004_0002_00000000.jpg")
    _image(images_dir / "00000004_0003_00000000.jpg")
    _pickle(
        tmp_path / "duke" / "partitions.pkl",
        {
            "val_im_names": [
                "00000004_0002_00000000.jpg",
                "00000004_0003_00000000.jpg",
            ],
            "val_marks": [0, 1],
        },
    )

    dataset = DukeProcessedTest(root=str(tmp_path), split="val")

    assert dataset.pids.tolist() == [4, 4]
    assert dataset.cams.tolist() == [2, 3]
    assert dataset.marks.tolist() == [0, 1]


def test_duke_processed_rejects_invalid_splits(tmp_path):
    with pytest.raises(ValueError, match="Duke processed train supports"):
        DukeProcessedTrain(root=str(tmp_path), split="test")

    with pytest.raises(ValueError, match="Duke processed test supports"):
        DukeProcessedTest(root=str(tmp_path), split="train")


def test_duke_processed_train_requires_images_and_partitions(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    part_file = tmp_path / "duke" / "partitions.pkl"

    with pytest.raises(FileNotFoundError, match=str(images_dir)):
        DukeProcessedTrain(root=str(tmp_path), split="trainval")

    images_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=str(part_file)):
        DukeProcessedTrain(root=str(tmp_path), split="trainval")


def test_duke_processed_test_requires_images_and_partitions(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    part_file = tmp_path / "duke" / "partitions.pkl"

    with pytest.raises(FileNotFoundError, match=str(images_dir)):
        DukeProcessedTest(root=str(tmp_path), split="test")

    images_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=str(part_file)):
        DukeProcessedTest(root=str(tmp_path), split="test")


def test_duke_processed_train_requires_expected_partition_keys(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    _image(images_dir / "00000003_0001_00000000.jpg")
    _pickle(
        tmp_path / "duke" / "partitions.pkl",
        {
            "trainval_im_names": ["00000003_0001_00000000.jpg"],
        },
    )

    with pytest.raises(KeyError, match="trainval_ids2labels"):
        DukeProcessedTrain(root=str(tmp_path), split="trainval")


def test_duke_processed_test_requires_expected_partition_keys(tmp_path):
    images_dir = tmp_path / "duke" / "images"
    _image(images_dir / "00000004_0002_00000000.jpg")
    _pickle(
        tmp_path / "duke" / "partitions.pkl",
        {
            "test_im_names": ["00000004_0002_00000000.jpg"],
        },
    )

    with pytest.raises(KeyError, match="test_marks"):
        DukeProcessedTest(root=str(tmp_path), split="test")
