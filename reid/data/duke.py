import glob
import os.path as osp
import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from reid.data.cuhk03 import parse_processed_reid_name
from reid.data.protocol import ReIDEvalSample, ReIDTrainSample
from reid.utils.io import expand, load_pickle


DUKE_RAW_PATTERN = re.compile(r"([-\d]+)_c(\d)")


def parse_duke_raw_name(name: str) -> Tuple[int, int]:
    base = osp.basename(name)
    match = DUKE_RAW_PATTERN.search(base)
    if match is None:
        raise ValueError(f"Cannot parse Duke pid/camid from: {name}")
    pid = int(match.group(1))
    camid = int(match.group(2))
    assert 1 <= camid <= 8, f"Invalid Duke camid {camid} in {name}"
    return pid, camid - 1


def parse_duke_name(name: str) -> Tuple[int, int]:
    return parse_duke_raw_name(name)


def parse_raw_duke_dir(dir_path: str, relabel: bool) -> List[Tuple[str, int, int]]:
    dir_path = expand(dir_path)
    if not osp.isdir(dir_path):
        raise FileNotFoundError(dir_path)

    img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
    pid_container = set()
    for img_path in img_paths:
        pid, _ = parse_duke_raw_name(img_path)
        if pid == -1:
            continue
        pid_container.add(pid)

    pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
    records: List[Tuple[str, int, int]] = []
    for img_path in img_paths:
        pid, camid = parse_duke_raw_name(img_path)
        if pid == -1:
            continue
        if relabel:
            pid = pid2label[pid]
        records.append((img_path, pid, camid))
    return records


def _normalize_names(names) -> List[str]:
    return [
        name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
        for name in names
    ]


def _require_partition_keys(parts, part_file: str, keys: List[str]) -> None:
    missing = [key for key in keys if key not in parts]
    if missing:
        raise KeyError(
            f"Duke partitions file {part_file} is missing keys: {missing}. "
            f"Available keys={list(parts.keys())}"
        )


class DukeRawTrain(Dataset):
    """
    Raw DukeMTMC-ReID train split:
      root/duke/DukeMTMC-reID/bounding_box_train/
    """

    def __init__(self, root: str, split: str = "trainval", transform=None):
        if split not in {"train", "trainval"}:
            raise ValueError(f"Duke raw train supports split 'train' or 'trainval', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform
        self.im_dir = osp.join(self.root, "duke", "DukeMTMC-reID", "bounding_box_train")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(self.im_dir)

        raw_records = parse_raw_duke_dir(self.im_dir, relabel=False)
        unique_pids = sorted({pid for _, pid, _ in raw_records})
        self.pid2label = {pid: label for label, pid in enumerate(unique_pids)}
        self.samples: List[Tuple[str, int, int, int]] = [
            (img_path, pid, camid, self.pid2label[pid])
            for img_path, pid, camid in raw_records
        ]
        self.im_names = [osp.basename(img_path) for img_path, _, _, _ in self.samples]
        self.pids = [pid for _, pid, _, _ in self.samples]
        self.cams = [camid for _, _, camid, _ in self.samples]
        self.labels = [label for _, _, _, label in self.samples]
        self.num_classes = len(unique_pids)
        self.num_cameras = len({int(camid) for camid in self.cams})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReIDTrainSample:
        img_path, _, _, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDTrainSample(image=img, label=int(label))


class DukeRawTest(Dataset):
    """
    Raw DukeMTMC-ReID eval split:
      root/duke/DukeMTMC-reID/query/
      root/duke/DukeMTMC-reID/bounding_box_test/
    """

    def __init__(self, root: str, split: str = "test", transform=None):
        if split not in {"val", "test"}:
            raise ValueError(f"Duke raw test supports split 'val' or 'test', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform
        base_dir = osp.join(self.root, "duke", "DukeMTMC-reID")
        self.query_dir = osp.join(base_dir, "query")
        self.gallery_dir = osp.join(base_dir, "bounding_box_test")
        if not osp.isdir(self.query_dir):
            raise FileNotFoundError(self.query_dir)
        if not osp.isdir(self.gallery_dir):
            raise FileNotFoundError(self.gallery_dir)

        query_records = parse_raw_duke_dir(self.query_dir, relabel=False)
        gallery_records = parse_raw_duke_dir(self.gallery_dir, relabel=False)
        self.samples = [
            (img_path, pid, camid, 0)
            for img_path, pid, camid in query_records
        ] + [
            (img_path, pid, camid, 1)
            for img_path, pid, camid in gallery_records
        ]

        self.im_names = [osp.basename(img_path) for img_path, _, _, _ in self.samples]
        self.pids = np.array([pid for _, pid, _, _ in self.samples], dtype=np.int64)
        self.cams = np.array([camid for _, _, camid, _ in self.samples], dtype=np.int64)
        self.marks = np.array([mark for _, _, _, mark in self.samples], dtype=np.int64)
        self.num_query = len(query_records)
        self.num_gallery = len(gallery_records)
        self.num_cameras = len({int(camid) for camid in self.cams})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReIDEvalSample:
        img_path, pid, camid, mark = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDEvalSample(
            image=img,
            pid=int(pid),
            camid=int(camid),
            image_name=osp.basename(img_path),
            mark=int(mark),
        )


class DukeProcessedTrain(Dataset):
    """
    Processed DukeMTMC-ReID train split:
      root/duke/images/
      root/duke/partitions.pkl
    """

    def __init__(self, root: str, split: str, transform=None):
        if split not in {"train", "trainval"}:
            raise ValueError(f"Duke processed train supports split 'train' or 'trainval', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform

        base_dir = osp.join(self.root, "duke")
        self.im_dir = osp.join(base_dir, "images")
        part_file = osp.join(base_dir, "partitions.pkl")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(self.im_dir)
        if not osp.isfile(part_file):
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
        self.num_cameras = len({int(camid) for camid in self.cams})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReIDTrainSample:
        name, _, _, label = self.samples[idx]
        img = Image.open(osp.join(self.im_dir, name)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDTrainSample(image=img, label=int(label))


class DukeProcessedTest(Dataset):
    """
    Processed DukeMTMC-ReID eval split:
      root/duke/images/
      root/duke/partitions.pkl
    """

    def __init__(self, root: str, split: str = "test", transform=None):
        if split not in {"val", "test"}:
            raise ValueError(f"Duke processed test supports split 'val' or 'test', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform

        base_dir = osp.join(self.root, "duke")
        self.im_dir = osp.join(base_dir, "images")
        part_file = osp.join(base_dir, "partitions.pkl")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(self.im_dir)
        if not osp.isfile(part_file):
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
        self.num_query = int(np.sum(self.marks == 0))
        self.num_gallery = int(np.sum(self.marks == 1))
        self.num_cameras = len({int(camid) for camid in self.cams})

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
