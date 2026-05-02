import pickle

import pytest
from PIL import Image

from reid.data.market1501 import (
    Market1501FromPartitions,
    Market1501ProcessedTrain,
    Market1501RawTrain,
    parse_market1501_name,
    parse_processed_name,
    parse_raw_market1501_dir,
)
from reid.data.market1501_test import (
    Market1501ProcessedTest,
    Market1501RawTest,
    Market1501TestFromPartitions,
)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 8), color=(255, 0, 0)).save(path)


def _pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)


def test_parse_market1501_name_returns_pid_and_zero_based_camid():
    pid, camid = parse_market1501_name("0002_c1s1_000451_01.jpg")

    assert pid == 2
    assert camid == 0


def test_parse_market1501_name_keeps_junk_pid():
    pid, camid = parse_market1501_name("-1_c1s1_000451_01.jpg")

    assert pid == -1
    assert camid == 0


def test_parse_processed_name_returns_transformed_pid_and_camid():
    pid, camid = parse_processed_name("00000002_0001_00000000.jpg")

    assert pid == 2
    assert camid == 1


def test_parse_raw_market1501_dir_relabels_train_pids_and_zero_bases_camids(tmp_path):
    raw_dir = tmp_path / "bounding_box_train"
    _touch(raw_dir / "0002_c1s1_000451_01.jpg")
    _touch(raw_dir / "0007_c6s1_000452_01.jpg")
    _touch(raw_dir / "0002_c3s1_000453_01.jpg")
    _touch(raw_dir / "-1_c1s1_000454_01.jpg")

    records = parse_raw_market1501_dir(str(raw_dir), relabel=True)

    assert records == [
        (str(raw_dir / "0002_c1s1_000451_01.jpg"), 0, 0),
        (str(raw_dir / "0002_c3s1_000453_01.jpg"), 0, 2),
        (str(raw_dir / "0007_c6s1_000452_01.jpg"), 1, 5),
    ]


def test_processed_market1501_aliases_keep_existing_classes():
    assert Market1501ProcessedTrain is Market1501FromPartitions
    assert Market1501ProcessedTest is Market1501TestFromPartitions


def test_parse_raw_market1501_dir_keeps_eval_pids_unrelabeled(tmp_path):
    raw_dir = tmp_path / "query"
    _touch(raw_dir / "0002_c1s1_000451_01.jpg")
    _touch(raw_dir / "0007_c6s1_000452_01.jpg")

    records = parse_raw_market1501_dir(str(raw_dir), relabel=False)

    assert records == [
        (str(raw_dir / "0002_c1s1_000451_01.jpg"), 2, 0),
        (str(raw_dir / "0007_c6s1_000452_01.jpg"), 7, 5),
    ]


def test_parse_raw_market1501_dir_rejects_out_of_range_pid(tmp_path):
    raw_dir = tmp_path / "query"
    _touch(raw_dir / "1502_c1s1_000451_01.jpg")

    with pytest.raises(AssertionError, match="Invalid Market1501 pid"):
        parse_raw_market1501_dir(str(raw_dir), relabel=False)


def test_parse_raw_market1501_dir_rejects_out_of_range_camid(tmp_path):
    raw_dir = tmp_path / "query"
    _touch(raw_dir / "0002_c7s1_000451_01.jpg")

    with pytest.raises(AssertionError, match="Invalid Market1501 camid"):
        parse_raw_market1501_dir(str(raw_dir), relabel=False)


def test_market1501_raw_train_exposes_samples_labels_and_num_classes(tmp_path):
    train_dir = tmp_path / "market1501" / "Market-1501-v15.09.15" / "bounding_box_train"
    _image(train_dir / "0002_c1s1_000451_01.jpg")
    _image(train_dir / "0007_c6s1_000452_01.jpg")
    _image(train_dir / "0002_c3s1_000453_01.jpg")
    _image(train_dir / "-1_c1s1_000454_01.jpg")

    dataset = Market1501RawTrain(root=str(tmp_path), transform=lambda img: img)

    assert dataset.samples == [
        (str(train_dir / "0002_c1s1_000451_01.jpg"), 2, 0, 0),
        (str(train_dir / "0002_c3s1_000453_01.jpg"), 2, 2, 0),
        (str(train_dir / "0007_c6s1_000452_01.jpg"), 7, 5, 1),
    ]
    assert dataset.labels == [0, 0, 1]
    assert dataset.num_classes == 2

    img, label = dataset[0]

    assert img.size == (4, 8)
    assert label == 0


def test_market1501_processed_train_returns_image_label_and_num_classes(tmp_path):
    images_dir = tmp_path / "market1501" / "images"
    _image(images_dir / "00000002_0001_00000000.jpg")
    _image(images_dir / "00000007_0006_00000000.jpg")
    _pickle(
        tmp_path / "market1501" / "partitions.pkl",
        {
            "trainval_im_names": [
                "00000002_0001_00000000.jpg",
                "00000007_0006_00000000.jpg",
            ],
            "trainval_ids2labels": {2: 0, 7: 1},
        },
    )

    dataset = Market1501FromPartitions(root=str(tmp_path), split="trainval", transform=lambda img: img)

    assert dataset.labels == [0, 1]
    assert dataset.pids == [2, 7]
    assert dataset.cams == [1, 6]
    assert dataset.num_classes == 2

    img, label = dataset[1]

    assert img.size == (4, 8)
    assert label == 1


def test_market1501_raw_train_rejects_eval_split(tmp_path):
    with pytest.raises(ValueError, match="Raw Market1501 train supports"):
        Market1501RawTrain(root=str(tmp_path), split="test")


def test_market1501_raw_train_requires_train_directory(tmp_path):
    expected = tmp_path / "market1501" / "Market-1501-v15.09.15" / "bounding_box_train"

    with pytest.raises(FileNotFoundError, match=str(expected)):
        Market1501RawTrain(root=str(tmp_path))


def test_market1501_raw_test_keeps_query_first_then_gallery(tmp_path):
    raw_root = tmp_path / "market1501" / "Market-1501-v15.09.15"
    query_dir = raw_root / "query"
    gallery_dir = raw_root / "bounding_box_test"
    _image(query_dir / "0002_c1s1_000451_01.jpg")
    _image(query_dir / "0007_c6s1_000452_01.jpg")
    _image(gallery_dir / "0002_c2s1_000453_01.jpg")
    _image(gallery_dir / "0008_c5s1_000454_01.jpg")
    _image(gallery_dir / "-1_c1s1_000455_01.jpg")

    dataset = Market1501RawTest(root=str(tmp_path), transform=lambda img: img)

    assert dataset.samples == [
        (str(query_dir / "0002_c1s1_000451_01.jpg"), 2, 0, 0),
        (str(query_dir / "0007_c6s1_000452_01.jpg"), 7, 5, 0),
        (str(gallery_dir / "0002_c2s1_000453_01.jpg"), 2, 1, 1),
        (str(gallery_dir / "0008_c5s1_000454_01.jpg"), 8, 4, 1),
    ]
    assert dataset.im_names == [
        "0002_c1s1_000451_01.jpg",
        "0007_c6s1_000452_01.jpg",
        "0002_c2s1_000453_01.jpg",
        "0008_c5s1_000454_01.jpg",
    ]
    assert dataset.pids.tolist() == [2, 7, 2, 8]
    assert dataset.cams.tolist() == [0, 5, 1, 4]
    assert dataset.marks.tolist() == [0, 0, 1, 1]

    img, pid, camid, image_name, mark = dataset[2]

    assert img.size == (4, 8)
    assert pid == 2
    assert camid == 1
    assert image_name == "0002_c2s1_000453_01.jpg"
    assert mark == 1


def test_market1501_raw_test_rejects_train_split(tmp_path):
    with pytest.raises(ValueError, match="Raw Market1501 test supports"):
        Market1501RawTest(root=str(tmp_path), split="train")


def test_market1501_processed_test_returns_eval_tuple(tmp_path):
    images_dir = tmp_path / "market1501" / "images"
    _image(images_dir / "00000002_0001_00000000.jpg")
    _image(images_dir / "00000002_0002_00000000.jpg")
    _pickle(
        tmp_path / "market1501" / "partitions.pkl",
        {
            "test_im_names": [
                "00000002_0001_00000000.jpg",
                "00000002_0002_00000000.jpg",
            ],
            "test_marks": [0, 1],
        },
    )

    dataset = Market1501TestFromPartitions(root=str(tmp_path), split="test", transform=lambda img: img)

    assert dataset.pids.tolist() == [2, 2]
    assert dataset.cams.tolist() == [1, 2]
    assert dataset.marks.tolist() == [0, 1]

    img, pid, camid, image_name, mark = dataset[0]

    assert img.size == (4, 8)
    assert pid == 2
    assert camid == 1
    assert image_name == "00000002_0001_00000000.jpg"
    assert mark == 0


def test_market1501_raw_test_requires_query_and_gallery_directories(tmp_path):
    raw_root = tmp_path / "market1501" / "Market-1501-v15.09.15"
    gallery_dir = raw_root / "bounding_box_test"
    gallery_dir.mkdir(parents=True)
    query_dir = raw_root / "query"

    with pytest.raises(FileNotFoundError, match=str(query_dir)):
        Market1501RawTest(root=str(tmp_path))

    query_dir.mkdir(parents=True)
    gallery_dir.rmdir()

    with pytest.raises(FileNotFoundError, match=str(gallery_dir)):
        Market1501RawTest(root=str(tmp_path))


def test_market1501_processed_train_requires_images_and_partitions(tmp_path):
    from reid.data.market1501 import Market1501FromPartitions

    images_dir = tmp_path / "market1501" / "images"
    part_file = tmp_path / "market1501" / "partitions.pkl"

    with pytest.raises(FileNotFoundError, match=str(images_dir)):
        Market1501FromPartitions(root=str(tmp_path), split="trainval")

    images_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=str(part_file)):
        Market1501FromPartitions(root=str(tmp_path), split="trainval")


def test_market1501_processed_test_requires_images_and_partitions(tmp_path):
    from reid.data.market1501_test import Market1501TestFromPartitions

    images_dir = tmp_path / "market1501" / "images"
    part_file = tmp_path / "market1501" / "partitions.pkl"

    with pytest.raises(FileNotFoundError, match=str(images_dir)):
        Market1501TestFromPartitions(root=str(tmp_path), split="test")

    images_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=str(part_file)):
        Market1501TestFromPartitions(root=str(tmp_path), split="test")
