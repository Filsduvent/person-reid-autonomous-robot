import os.path as osp
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from reid.utils.io import load_pickle, expand

def parse_id_from_name(name: str) -> int:
    """
    Supports:
      - transformed naming: '00000012_0003_00000000.jpg'  -> id=12
      - original Market1501: '0002_c1s1_000451_01.jpg'   -> id=2
    """
    base = osp.basename(name)
    if "_" in base and base[:8].isdigit():
        return int(base[:8])
    # original market: first 4 chars are id, can be -1
    pid = base[:4]
    if pid == "-1":
        return -1
    if pid.isdigit():
        return int(pid)
    raise ValueError(f"Cannot parse id from: {name}")

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
        kept = [(n, pid, lab) for n, pid, lab in zip(self.im_names, [parse_id_from_name(x) for x in self.im_names], [self.ids2labels.get(parse_id_from_name(x), -1) for x in self.im_names]) if pid >= 0]
        self.im_names = [x[0] for x in kept]
        self.pids = [x[1] for x in kept]
        self.labels = [x[2] for x in kept]

    def __len__(self) -> int:
        return len(self.im_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        name = self.im_names[idx]
        path = osp.join(self.im_dir, name)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
