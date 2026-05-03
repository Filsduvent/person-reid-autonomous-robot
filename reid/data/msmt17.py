import os.path as osp
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from reid.data.protocol import ReIDEvalSample, ReIDTrainSample
from reid.utils.io import expand


def _parse_msmt17_camid(rel_path: str) -> int:
    """
    MSMT17 official code extracts camera id with:
      int(img_path.split("_")[2])

    Keep that convention, but validate the token so malformed list entries fail
    with a clear error instead of an IndexError or a vague ValueError.
    """
    parts = rel_path.split("_")
    if len(parts) <= 2 or not parts[2].isdigit():
        raise ValueError(
            "Cannot parse MSMT17 camera id from list image path. "
            "Expected an underscore-delimited path with a numeric third token, "
            f"got: {rel_path}"
        )
    return int(parts[2])


def parse_msmt17_list(list_path: str, image_dir: str) -> List[Tuple[str, int, int]]:
    """
    Parse an official MSMT17 list file into (img_path, pid, camid).

    Official MSMT17 list rows are:
      relative/image/path.jpg pid

    MSMT17 camera ids are read from the third underscore-delimited token, matching
    the original implementation. This pipeline uses zero-based camera ids, so the
    parser normalizes one-based lists by subtracting one when the minimum observed
    camera id is 1. Lists that already start at 0 are kept unchanged.
    """
    list_path = expand(list_path)
    image_dir = expand(image_dir)
    if not osp.isfile(list_path):
        raise FileNotFoundError(list_path)
    if not osp.isdir(image_dir):
        raise FileNotFoundError(image_dir)

    rows = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"MSMT17 list row must contain image path and pid, "
                    f"got {len(parts)} fields at {list_path}:{line_num}: {line}"
                )
            rel_path, pid_text = parts
            try:
                pid = int(pid_text)
            except ValueError as exc:
                raise ValueError(
                    f"MSMT17 pid must be an integer at {list_path}:{line_num}: {pid_text}"
                ) from exc
            camid = _parse_msmt17_camid(rel_path)
            rows.append((osp.join(image_dir, rel_path), pid, camid))

    if not rows:
        return []

    min_camid = min(camid for _, _, camid in rows)
    if min_camid < 0:
        raise ValueError(f"MSMT17 camera ids must be non-negative in {list_path}.")
    if min_camid == 1:
        return [(img_path, pid, camid - 1) for img_path, pid, camid in rows]
    if min_camid == 0:
        return rows

    raise ValueError(
        "Cannot infer MSMT17 camera-id base from list because the minimum "
        f"camera id is {min_camid}. Expected minimum 0 or 1 in {list_path}."
    )


class MSMT17RawTrain(Dataset):
    """
    Official raw MSMT17 train split:
      root/msmt17/MSMT17_V2/mask_train_v2/
      root/msmt17/MSMT17_V2/list_train.txt
      root/msmt17/MSMT17_V2/list_val.txt
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        if split not in {"train", "val", "trainval"}:
            raise ValueError(f"MSMT17 raw train supports split 'train', 'val', or 'trainval', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform
        self.dataset_dir = osp.join(self.root, "msmt17", "MSMT17_V2")
        if not osp.isdir(self.dataset_dir):
            raise FileNotFoundError(self.dataset_dir)
        self.im_dir = osp.join(self.dataset_dir, "mask_train_v2")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(self.im_dir)

        list_names = {
            "train": ["list_train.txt"],
            "val": ["list_val.txt"],
            "trainval": ["list_train.txt", "list_val.txt"],
        }[split]

        records: List[Tuple[str, int, int]] = []
        for list_name in list_names:
            records.extend(parse_msmt17_list(osp.join(self.dataset_dir, list_name), self.im_dir))

        unique_pids = sorted({pid for _, pid, _ in records})
        self.pid2label = {pid: label for label, pid in enumerate(unique_pids)}
        self.samples: List[Tuple[str, int, int, int]] = [
            (img_path, pid, camid, self.pid2label[pid])
            for img_path, pid, camid in records
        ]
        self.im_names = [osp.relpath(img_path, self.im_dir) for img_path, _, _, _ in self.samples]
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


class MSMT17RawTest(Dataset):
    """
    Official raw MSMT17 eval split:
      root/msmt17/MSMT17_V2/mask_test_v2/
      root/msmt17/MSMT17_V2/list_query.txt
      root/msmt17/MSMT17_V2/list_gallery.txt
    """

    def __init__(self, root: str, split: str = "test", transform=None):
        if split != "test":
            raise ValueError(f"MSMT17 raw test supports split 'test', got '{split}'.")

        self.root = expand(root)
        self.split = split
        self.transform = transform
        self.dataset_dir = osp.join(self.root, "msmt17", "MSMT17_V2")
        if not osp.isdir(self.dataset_dir):
            raise FileNotFoundError(self.dataset_dir)
        self.im_dir = osp.join(self.dataset_dir, "mask_test_v2")
        if not osp.isdir(self.im_dir):
            raise FileNotFoundError(self.im_dir)

        query_records = parse_msmt17_list(osp.join(self.dataset_dir, "list_query.txt"), self.im_dir)
        gallery_records = parse_msmt17_list(osp.join(self.dataset_dir, "list_gallery.txt"), self.im_dir)
        self.samples: List[Tuple[str, int, int, int]] = [
            (img_path, pid, camid, 0)
            for img_path, pid, camid in query_records
        ] + [
            (img_path, pid, camid, 1)
            for img_path, pid, camid in gallery_records
        ]

        self.im_names = [osp.relpath(img_path, self.im_dir) for img_path, _, _, _ in self.samples]
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
            image_name=osp.relpath(img_path, self.im_dir),
            mark=int(mark),
        )
