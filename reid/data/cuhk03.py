import os.path as osp
import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from reid.data.protocol import ReIDEvalSample, ReIDTrainSample
from reid.utils.io import expand, load_pickle


VALID_IMAGE_TYPES = {"detected", "labeled"}
PROCESSED_PARTITION_PROTOCOL = "processed_partition"
PROCESSED_REID_PATTERN = re.compile(r"^(\d{8})_(\d{4})_(\d{8})\.(jpg|png)$", re.IGNORECASE)

# TODO: Add raw CUHK03 .mat preprocessing support later if needed.


def parse_processed_reid_name(name: str) -> Tuple[int, int]:
    """
    Parse transformed ReID names using the shared processed convention:
      00000002_0001_00000000.jpg -> pid=2, camid=1

    The transformed tri_loss-style files store camera in name[9:13].
    Keep int(name[9:13]) unchanged for compatibility with existing partitions.
    """
    base = osp.basename(name)
    match = PROCESSED_REID_PATTERN.match(base)
    if match is None:
        raise ValueError(
            "Processed ReID filename must match "
            "8digits_4digits_8digits.ext with .jpg or .png extension, "
            f"got: {name}"
        )
    return int(match.group(1)), int(match.group(2))


def parse_processed_name(name: str) -> Tuple[int, int]:
    return parse_processed_reid_name(name)


def _normalize_names(names) -> List[str]:
    return [
        name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
        for name in names
    ]


def _validate_image_type(image_type: str) -> str:
    image_type = str(image_type).lower()
    if image_type not in VALID_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported CUHK03 image_type '{image_type}'. "
            "Use 'detected' or 'labeled'."
        )
    return image_type


def _validate_protocol(protocol: str, split_id: int | None) -> str:
    """Validate the protocol represented by the available processed files.

    The repository's processed CUHK03 partitions are flat files containing
    train/val/test lists.  They contain neither a semantic protocol label nor
    split identifier, so claiming ``new`` or ``classic`` would be unverifiable.
    ``processed_partition`` makes that contract explicit.  A future converter
    can add a protocol-aware loader without changing the common data API.
    """
    value = str(protocol).lower()
    if value != PROCESSED_PARTITION_PROTOCOL:
        raise ValueError(
            "CUHK03 processed partitions do not encode 'new' or 'classic' "
            "protocol metadata. Use protocol='processed_partition' until a "
            "protocol-aware conversion is supplied."
        )
    if split_id is not None:
        raise ValueError(
            "CUHK03 split_id is not represented in flat processed partitions; "
            "set split_id to null."
        )
    return value


def _require_partition_keys(parts, part_file: str, keys: List[str]) -> None:
    missing = [key for key in keys if key not in parts]
    if missing:
        raise KeyError(
            f"CUHK03 partitions file {part_file} is missing keys: {missing}. "
            f"Available keys={list(parts.keys())}"
        )


class CUHK03ProcessedTrain(Dataset):
    """
    Processed CUHK03 train split:
      root/cuhk03/{detected,labeled}/images/
      root/cuhk03/{detected,labeled}/partitions.pkl
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_type: str = "detected",
        protocol: str = PROCESSED_PARTITION_PROTOCOL,
        split_id: int | None = None,
        transform=None,
    ):
        if split not in {"train", "trainval"}:
            raise ValueError(f"CUHK03 processed train supports split 'train' or 'trainval', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.image_type = _validate_image_type(image_type)
        self.protocol = _validate_protocol(protocol, split_id)
        self.split_id = split_id
        self.transform = transform

        base_dir = osp.join(self.root, "cuhk03", self.image_type)
        self.im_dir = osp.join(base_dir, "images")
        part_file = osp.join(base_dir, "partitions.pkl")
        if not osp.exists(self.im_dir):
            raise FileNotFoundError(self.im_dir)
        if not osp.exists(part_file):
            raise FileNotFoundError(part_file)

        parts = load_pickle(part_file)
        key = f"{split}_im_names"
        label_key = f"{split}_ids2labels"
        _require_partition_keys(parts, part_file, [key, label_key])

        im_names = _normalize_names(parts[key])
        ids2labels: Dict[int, int] = parts[label_key]
        self.ids2labels = ids2labels

        self.samples: List[Tuple[str, int, int, int]] = []
        for name in im_names:
            pid, camid = parse_processed_reid_name(name)
            if pid < 0:
                continue
            self.samples.append((name, pid, camid, int(self.ids2labels[pid])))

        self.im_names = [name for name, _, _, _ in self.samples]
        self.pids = [pid for _, pid, _, _ in self.samples]
        self.cams = [camid for _, _, camid, _ in self.samples]
        self.labels = [label for _, _, _, label in self.samples]
        self.num_classes = len(self.ids2labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReIDTrainSample:
        name, _, _, label = self.samples[idx]
        img = Image.open(osp.join(self.im_dir, name)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDTrainSample(image=img, label=int(label))


class CUHK03ProcessedTest(Dataset):
    """
    Processed CUHK03 eval split:
      root/cuhk03/{detected,labeled}/images/
      root/cuhk03/{detected,labeled}/partitions.pkl
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        image_type: str = "detected",
        protocol: str = PROCESSED_PARTITION_PROTOCOL,
        split_id: int | None = None,
        transform=None,
    ):
        if split not in {"val", "test"}:
            raise ValueError(f"CUHK03 processed test supports split 'val' or 'test', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.image_type = _validate_image_type(image_type)
        self.protocol = _validate_protocol(protocol, split_id)
        self.split_id = split_id
        self.transform = transform

        base_dir = osp.join(self.root, "cuhk03", self.image_type)
        self.im_dir = osp.join(base_dir, "images")
        part_file = osp.join(base_dir, "partitions.pkl")
        if not osp.exists(self.im_dir):
            raise FileNotFoundError(self.im_dir)
        if not osp.exists(part_file):
            raise FileNotFoundError(part_file)

        parts = load_pickle(part_file)
        key_names = f"{split}_im_names"
        key_marks = f"{split}_marks"
        _require_partition_keys(parts, part_file, [key_names, key_marks])

        self.im_names = np.array(_normalize_names(parts[key_names]))
        self.marks = np.array(parts[key_marks], dtype=np.int64)

        pids, cams = [], []
        for name in self.im_names:
            pid, camid = parse_processed_reid_name(name)
            pids.append(pid)
            cams.append(camid)
        self.pids = np.array(pids, dtype=np.int64)
        self.cams = np.array(cams, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.im_names)

    def __getitem__(self, idx: int) -> ReIDEvalSample:
        name = str(self.im_names[idx])
        img = Image.open(osp.join(self.im_dir, name)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDEvalSample(
            image=img,
            pid=int(self.pids[idx]),
            camid=int(self.cams[idx]),
            image_name=name,
            mark=int(self.marks[idx]),
        )
