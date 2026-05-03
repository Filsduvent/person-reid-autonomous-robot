import pytest
from PIL import Image

from reid.data.msmt17 import MSMT17RawTest, MSMT17RawTrain, parse_msmt17_list


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 8), color=(255, 0, 0)).save(path)


def _write_list(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_msmt17_list_reads_pid_and_zero_bases_one_based_camids(tmp_path):
    image_dir = tmp_path / "msmt17" / "MSMT17_V2" / "mask_train_v2"
    image_dir.mkdir(parents=True)
    list_path = tmp_path / "msmt17" / "MSMT17_V2" / "list_train.txt"
    _write_list(
        list_path,
        [
            "0000/0000_0000_1_0001.jpg 0",
            "0001/0001_0001_3_0002.jpg 1",
        ],
    )

    records = parse_msmt17_list(str(list_path), str(image_dir))

    assert records == [
        (str(image_dir / "0000/0000_0000_1_0001.jpg"), 0, 0),
        (str(image_dir / "0001/0001_0001_3_0002.jpg"), 1, 2),
    ]


def test_parse_msmt17_list_supports_synthetic_line_contract(tmp_path):
    image_dir = tmp_path / "msmt17" / "MSMT17_V2" / "mask_train_v2"
    image_dir.mkdir(parents=True)
    list_path = tmp_path / "msmt17" / "MSMT17_V2" / "list_train.txt"
    rel_path = "0000/0000_00_01_0001.jpg"
    _write_list(list_path, [f"{rel_path} 0"])

    records = parse_msmt17_list(str(list_path), str(image_dir))

    assert len(records) == 1
    sample_path, pid, camid = records[0]
    assert sample_path == str(image_dir / rel_path)
    assert pid == 0
    assert camid == 0
    assert isinstance(pid, int)
    assert isinstance(camid, int)


def test_parse_msmt17_list_keeps_already_zero_based_camids(tmp_path):
    image_dir = tmp_path / "msmt17" / "MSMT17_V2" / "mask_test_v2"
    image_dir.mkdir(parents=True)
    list_path = tmp_path / "msmt17" / "MSMT17_V2" / "list_query.txt"
    _write_list(
        list_path,
        [
            "0000/0000_0000_0_0001.jpg 0",
            "0001/0001_0001_2_0002.jpg 1",
        ],
    )

    records = parse_msmt17_list(str(list_path), str(image_dir))

    assert records == [
        (str(image_dir / "0000/0000_0000_0_0001.jpg"), 0, 0),
        (str(image_dir / "0001/0001_0001_2_0002.jpg"), 1, 2),
    ]


def test_parse_msmt17_list_requires_expected_paths(tmp_path):
    image_dir = tmp_path / "msmt17" / "MSMT17_V2" / "mask_train_v2"
    list_path = tmp_path / "msmt17" / "MSMT17_V2" / "list_train.txt"

    with pytest.raises(FileNotFoundError, match=str(list_path)):
        parse_msmt17_list(str(list_path), str(image_dir))

    _write_list(list_path, ["0000/0000_0000_1_0001.jpg 0"])
    with pytest.raises(FileNotFoundError, match=str(image_dir)):
        parse_msmt17_list(str(list_path), str(image_dir))


def test_parse_msmt17_list_rejects_malformed_rows(tmp_path):
    image_dir = tmp_path / "msmt17" / "MSMT17_V2" / "mask_train_v2"
    image_dir.mkdir(parents=True)
    list_path = tmp_path / "msmt17" / "MSMT17_V2" / "list_train.txt"

    _write_list(list_path, ["0000/0000_0000_1_0001.jpg"])
    with pytest.raises(ValueError, match="image path and pid"):
        parse_msmt17_list(str(list_path), str(image_dir))

    _write_list(list_path, ["0000/0000_0000_1_0001.jpg abc"])
    with pytest.raises(ValueError, match="pid must be an integer"):
        parse_msmt17_list(str(list_path), str(image_dir))

    _write_list(list_path, ["0000/bad_name.jpg 0"])
    with pytest.raises(ValueError, match="Cannot parse MSMT17 camera id"):
        parse_msmt17_list(str(list_path), str(image_dir))


def test_msmt17_raw_train_returns_image_label_and_metadata(tmp_path):
    dataset_dir = tmp_path / "msmt17" / "MSMT17_V2"
    image_dir = dataset_dir / "mask_train_v2"
    _image(image_dir / "0000/0000_0000_1_0001.jpg")
    _image(image_dir / "0001/0001_0001_2_0002.jpg")
    _image(image_dir / "0000/0000_0000_3_0003.jpg")
    _write_list(
        dataset_dir / "list_train.txt",
        [
            "0000/0000_0000_1_0001.jpg 0",
            "0001/0001_0001_2_0002.jpg 1",
            "0000/0000_0000_3_0003.jpg 0",
        ],
    )

    dataset = MSMT17RawTrain(root=str(tmp_path), split="train", transform=lambda img: img)

    assert dataset.samples == [
        (str(image_dir / "0000/0000_0000_1_0001.jpg"), 0, 0, 0),
        (str(image_dir / "0001/0001_0001_2_0002.jpg"), 1, 1, 1),
        (str(image_dir / "0000/0000_0000_3_0003.jpg"), 0, 2, 0),
    ]
    assert dataset.im_names == [
        "0000/0000_0000_1_0001.jpg",
        "0001/0001_0001_2_0002.jpg",
        "0000/0000_0000_3_0003.jpg",
    ]
    assert dataset.labels == [0, 1, 0]
    assert dataset.num_classes == 2

    img, label = dataset[0]

    assert img.size == (4, 8)
    assert label == 0


def test_msmt17_raw_train_supports_val_and_trainval_splits(tmp_path):
    dataset_dir = tmp_path / "msmt17" / "MSMT17_V2"
    image_dir = dataset_dir / "mask_train_v2"
    _image(image_dir / "0000/0000_0000_1_0001.jpg")
    _image(image_dir / "0001/0001_0001_1_0002.jpg")
    _write_list(dataset_dir / "list_train.txt", ["0000/0000_0000_1_0001.jpg 0"])
    _write_list(dataset_dir / "list_val.txt", ["0001/0001_0001_1_0002.jpg 1"])

    val_dataset = MSMT17RawTrain(root=str(tmp_path), split="val")
    trainval_dataset = MSMT17RawTrain(root=str(tmp_path), split="trainval")

    assert val_dataset.labels == [0]
    assert val_dataset.pids == [1]
    assert val_dataset.num_classes == 1
    assert trainval_dataset.labels == [0, 1]
    assert trainval_dataset.pids == [0, 1]
    assert trainval_dataset.num_classes == 2


def test_msmt17_raw_train_requires_expected_paths(tmp_path):
    dataset_dir = tmp_path / "msmt17" / "MSMT17_V2"
    image_dir = dataset_dir / "mask_train_v2"
    list_train = dataset_dir / "list_train.txt"
    list_val = dataset_dir / "list_val.txt"

    with pytest.raises(FileNotFoundError, match=str(dataset_dir)):
        MSMT17RawTrain(root=str(tmp_path), split="train")

    dataset_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=str(image_dir)):
        MSMT17RawTrain(root=str(tmp_path), split="train")

    image_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=str(list_train)):
        MSMT17RawTrain(root=str(tmp_path), split="train")

    _write_list(list_train, ["0000/0000_0000_1_0001.jpg 0"])
    with pytest.raises(FileNotFoundError, match=str(list_val)):
        MSMT17RawTrain(root=str(tmp_path), split="val")

    with pytest.raises(FileNotFoundError, match=str(list_val)):
        MSMT17RawTrain(root=str(tmp_path), split="trainval")


def test_msmt17_raw_train_rejects_invalid_split(tmp_path):
    with pytest.raises(ValueError, match="MSMT17 raw train supports"):
        MSMT17RawTrain(root=str(tmp_path), split="test")


def test_msmt17_raw_test_keeps_query_first_then_gallery(tmp_path):
    dataset_dir = tmp_path / "msmt17" / "MSMT17_V2"
    image_dir = dataset_dir / "mask_test_v2"
    _image(image_dir / "0000/0000_0000_1_0001.jpg")
    _image(image_dir / "0001/0001_0001_2_0002.jpg")
    _image(image_dir / "0000/0000_0000_1_0003.jpg")
    _image(image_dir / "0002/0002_0002_4_0004.jpg")
    _write_list(
        dataset_dir / "list_query.txt",
        [
            "0000/0000_0000_1_0001.jpg 0",
            "0001/0001_0001_2_0002.jpg 1",
        ],
    )
    _write_list(
        dataset_dir / "list_gallery.txt",
        [
            "0000/0000_0000_1_0003.jpg 0",
            "0002/0002_0002_4_0004.jpg 2",
        ],
    )

    dataset = MSMT17RawTest(root=str(tmp_path), split="test", transform=lambda img: img)

    assert dataset.samples == [
        (str(image_dir / "0000/0000_0000_1_0001.jpg"), 0, 0, 0),
        (str(image_dir / "0001/0001_0001_2_0002.jpg"), 1, 1, 0),
        (str(image_dir / "0000/0000_0000_1_0003.jpg"), 0, 0, 1),
        (str(image_dir / "0002/0002_0002_4_0004.jpg"), 2, 3, 1),
    ]
    assert dataset.im_names == [
        "0000/0000_0000_1_0001.jpg",
        "0001/0001_0001_2_0002.jpg",
        "0000/0000_0000_1_0003.jpg",
        "0002/0002_0002_4_0004.jpg",
    ]
    assert dataset.pids.tolist() == [0, 1, 0, 2]
    assert dataset.cams.tolist() == [0, 1, 0, 3]
    assert dataset.marks.tolist() == [0, 0, 1, 1]
    assert dataset.num_query == 2
    assert dataset.num_gallery == 2

    img, pid, camid, image_name, mark = dataset[2]

    assert img.size == (4, 8)
    assert pid == 0
    assert camid == 0
    assert image_name == "0000/0000_0000_1_0003.jpg"
    assert mark == 1


def test_msmt17_raw_test_requires_expected_paths(tmp_path):
    dataset_dir = tmp_path / "msmt17" / "MSMT17_V2"
    image_dir = dataset_dir / "mask_test_v2"
    query_list = dataset_dir / "list_query.txt"
    gallery_list = dataset_dir / "list_gallery.txt"

    with pytest.raises(FileNotFoundError, match=str(dataset_dir)):
        MSMT17RawTest(root=str(tmp_path), split="test")

    dataset_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=str(image_dir)):
        MSMT17RawTest(root=str(tmp_path), split="test")

    image_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=str(query_list)):
        MSMT17RawTest(root=str(tmp_path), split="test")

    _write_list(query_list, ["0000/0000_0000_1_0001.jpg 0"])
    with pytest.raises(FileNotFoundError, match=str(gallery_list)):
        MSMT17RawTest(root=str(tmp_path), split="test")


def test_msmt17_raw_test_rejects_invalid_split(tmp_path):
    with pytest.raises(ValueError, match="MSMT17 raw test supports"):
        MSMT17RawTest(root=str(tmp_path), split="val")
