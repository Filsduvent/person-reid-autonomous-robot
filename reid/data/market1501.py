import glob
import os.path as osp
import re
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from reid.data.protocol import ReIDTrainSample
from reid.utils.io import load_pickle, expand

MARKET1501_RAW_PATTERN = re.compile(r"([-\d]+)_c(\d)")


def parse_market1501_name(name: str) -> Tuple[int, int]:
    base = osp.basename(name)
    match = MARKET1501_RAW_PATTERN.search(base)
    if match is None:
        raise ValueError(f"Cannot parse Market1501 pid/camid from: {name}")
    pid = int(match.group(1))
    camid = int(match.group(2))
    if pid != -1:
        assert 0 <= pid <= 1501, f"Invalid Market1501 pid {pid} in {name}"
    assert 1 <= camid <= 6, f"Invalid Market1501 camid {camid} in {name}"
    return pid, camid - 1


def parse_processed_name(name: str) -> Tuple[int, int]:
    base = osp.basename(name)
    if len(base) >= 13 and base[:8].isdigit() and base[8] == "_" and base[9:13].isdigit():
        return int(base[:8]), int(base[9:13])
    return parse_market1501_name(base)


def parse_id_from_name(name: str) -> int:
    """
    Supports:
      - transformed naming: '00000012_0003_00000000.jpg'  -> id=12
      - original Market1501: '0002_c1s1_000451_01.jpg'   -> id=2
    """
    base = osp.basename(name)
    if "_" in base and base[:8].isdigit():
        pid, _ = parse_processed_name(base)
        return pid
    pid, _ = parse_market1501_name(base)
    return pid


def parse_cam_from_name(name: str) -> int:
    _, camid = parse_processed_name(name)
    return camid


def parse_raw_market1501_dir(dir_path: str, relabel: bool) -> List[Tuple[str, int, int]]:
    """
    Parse a standard Market1501 directory into (img_path, pid_or_label, camid).

    The raw directory format is used by:
      - bounding_box_train/ with relabel=True
      - query/ and bounding_box_test/ with relabel=False

    Raw camera ids are one-based in filenames and are converted to zero-based ids.
    """
    dir_path = expand(dir_path)
    if not osp.isdir(dir_path):
        raise FileNotFoundError(f"Market1501 raw directory not found: {dir_path}")

    img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
    pid_container = set()

    for img_path in img_paths:
        pid, _ = parse_market1501_name(img_path)
        if pid == -1:
            continue
        pid_container.add(pid)

    pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
    records: List[Tuple[str, int, int]] = []

    for img_path in img_paths:
        pid, camid = parse_market1501_name(img_path)
        if pid == -1:
            continue
        if relabel:
            pid = pid2label[pid]
        records.append((img_path, pid, camid))

    return records


class Market1501RawTrain(Dataset):
    """
    Standard raw Market1501 train split:
      root/market1501/Market-1501-v15.09.15/bounding_box_train/

    Exposes samples as (img_path, raw_pid, zero_based_camid, label), while
    __getitem__ follows the training pipeline contract: (image, label).
    """

    def __init__(self, root: str, split: str = "trainval", transform=None):
        if split not in {"train", "trainval"}:
            raise ValueError(f"Raw Market1501 train supports split 'train' or 'trainval', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform
        self.im_dir = osp.join(
            self.root,
            "market1501",
            "Market-1501-v15.09.15",
            "bounding_box_train",
        )
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(f"Market1501 raw train directory not found: {self.im_dir}")

        raw_records = parse_raw_market1501_dir(self.im_dir, relabel=False)
        unique_pids = sorted({pid for _, pid, _ in raw_records})
        self.pid2label = {pid: label for label, pid in enumerate(unique_pids)}
        self.samples: List[Tuple[str, int, int, int]] = [
            (img_path, pid, camid, self.pid2label[pid])
            for img_path, pid, camid in raw_records
        ]
        self.labels = [label for _, _, _, label in self.samples]
        self.pids = [pid for _, pid, _, _ in self.samples]
        self.cams = [camid for _, _, camid, _ in self.samples]
        self.im_names = [osp.basename(img_path) for img_path, _, _, _ in self.samples]
        self.num_classes = len(unique_pids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ReIDTrainSample:
        img_path, _, _, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDTrainSample(image=img, label=int(label))


class Market1501FromPartitions(Dataset):
    """
    Uses the transformed dataset layout used by your tri_loss scripts:
      root/market1501/images/
      root/market1501/partitions.pkl
    partitions keys typically include: trainval_im_names, train_im_names, etc.
    """
    def __init__(self, root: str, split: str, transform=None):
        self.root = expand(root)
        self.split = split  # "train" or "trainval"
        self.transform = transform

        base_dir = osp.join(self.root, "market1501")
        self.im_dir = osp.join(base_dir, "images")
        part_file = osp.join(base_dir, "partitions.pkl")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(f"Market1501 processed images directory not found: {self.im_dir}")
        if not osp.isfile(part_file):
            raise FileNotFoundError(f"Market1501 processed partitions file not found: {part_file}")
        parts = load_pickle(part_file)

        key = f"{split}_im_names"
        if key not in parts:
            raise KeyError(f"'{key}' not found in {part_file}. Keys={list(parts.keys())}")

        im_names = parts[key]
        # partitions may store bytes; normalize to str
        self.im_names: List[str] = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in im_names
        ]

        # label mapping if provided (preferred), else build from parsed ids
        ids2labels: Dict[int, int] = parts.get(f"{split}_ids2labels") or parts.get("trainval_ids2labels")
        if ids2labels is None:
            unique_ids = sorted({parse_id_from_name(n) for n in self.im_names if parse_id_from_name(n) >= 0})
            ids2labels = {pid: i for i, pid in enumerate(unique_ids)}
        self.ids2labels = ids2labels

        self.labels: List[int] = []
        self.pids: List[int] = []
        for n in self.im_names:
            pid = parse_id_from_name(n)
            if pid < 0:
                # skip junk id
                continue
            self.pids.append(pid)
            self.labels.append(self.ids2labels[pid])

        # keep aligned arrays after filtering
        kept = [
            (n, pid, parse_cam_from_name(n), lab)
            for n, pid, lab in zip(
                self.im_names,
                [parse_id_from_name(x) for x in self.im_names],
                [self.ids2labels.get(parse_id_from_name(x), -1) for x in self.im_names],
            )
            if pid >= 0
        ]
        self.im_names = [x[0] for x in kept]
        self.pids = [x[1] for x in kept]
        self.cams = [x[2] for x in kept]
        self.labels = [x[3] for x in kept]
        self.num_classes = len(set(int(x) for x in self.labels))

    def __len__(self) -> int:
        return len(self.im_names)

    def __getitem__(self, idx: int) -> ReIDTrainSample:
        name = self.im_names[idx]
        path = osp.join(self.im_dir, name)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[idx])
        return ReIDTrainSample(image=img, label=label)


Market1501ProcessedTrain = Market1501FromPartitions
