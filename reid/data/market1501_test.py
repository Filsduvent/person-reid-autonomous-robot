import os.path as osp
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from reid.data.market1501 import parse_processed_name, parse_raw_market1501_dir
from reid.data.protocol import ReIDEvalSample
from reid.utils.io import load_pickle, expand

def parse_pid_cam(name: str) -> Tuple[int, int]:
    return parse_processed_name(name)


class Market1501RawTest(Dataset):
    """
    Standard raw Market1501 eval split.

    Query samples are stored first with mark=0, gallery samples second with
    mark=1. This ordering is part of the evaluation contract.
    """

    def __init__(self, root: str, split: str = "test", transform=None):
        if split not in {"test", "val"}:
            raise ValueError(f"Raw Market1501 test supports split 'test' or 'val', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform
        base_dir = osp.join(self.root, "market1501", "Market-1501-v15.09.15")
        self.query_dir = osp.join(base_dir, "query")
        self.gallery_dir = osp.join(base_dir, "bounding_box_test")
        if not osp.isdir(self.query_dir):
            raise FileNotFoundError(f"Market1501 raw query directory not found: {self.query_dir}")
        if not osp.isdir(self.gallery_dir):
            raise FileNotFoundError(f"Market1501 raw gallery directory not found: {self.gallery_dir}")

        query_records = parse_raw_market1501_dir(self.query_dir, relabel=False)
        gallery_records = parse_raw_market1501_dir(self.gallery_dir, relabel=False)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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


class Market1501TestFromPartitions(Dataset):
    """
    Uses:
      root/market1501/images/
      root/market1501/partitions.pkl with keys:
        test_im_names, test_marks
    marks: 0=query, 1=gallery, 2=multi-query (optional)
    """
    def __init__(self, root: str, transform=None, split: str = "test"):
        self.root = expand(root)
        self.transform = transform
        base_dir = osp.join(self.root, "market1501")
        self.im_dir = osp.join(base_dir, "images")
        part_file = osp.join(base_dir, "partitions.pkl")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(f"Market1501 processed images directory not found: {self.im_dir}")
        if not osp.isfile(part_file):
            raise FileNotFoundError(f"Market1501 processed partitions file not found: {part_file}")
        parts = load_pickle(part_file)

        key_names = f"{split}_im_names"
        key_marks = f"{split}_marks"
        if key_names not in parts or key_marks not in parts:
            raise KeyError(f"Missing {key_names}/{key_marks} in partitions.pkl")

        im_names = parts[key_names]
        marks = parts[key_marks]

        self.im_names = np.array([n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in im_names])
        self.marks = np.array(marks, dtype=np.int64)

        # pid/cam arrays
        pids, cams = [], []
        for n in self.im_names:
            pid, cam = parse_pid_cam(n)
            pids.append(pid); cams.append(cam)
        self.pids = np.array(pids, dtype=np.int64)
        self.cams = np.array(cams, dtype=np.int64)

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        name = self.im_names[idx]
        path = osp.join(self.im_dir, name)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return ReIDEvalSample(
            image=img,
            pid=int(self.pids[idx]),
            camid=int(self.cams[idx]),
            image_name=str(name),
            mark=int(self.marks[idx]),
        )


Market1501ProcessedTest = Market1501TestFromPartitions
