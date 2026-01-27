import os.path as osp
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from reid.utils.io import load_pickle, expand

def parse_pid_cam(name: str) -> Tuple[int, int]:
    base = osp.basename(name)
    # transformed: 00000012_0003_00000000.jpg
    if len(base) >= 13 and base[:8].isdigit() and base[8] == "_" and base[9:13].isdigit():
        pid = int(base[:8])
        cam = int(base[9:13])
        return pid, cam
    # original market: 0002_c1s1_000451_01.jpg
    pid = -1 if base.startswith("-1") else int(base[:4])
    # cam is after 'c'
    cpos = base.find("c")
    cam = int(base[cpos + 1]) if cpos != -1 else 0
    return pid, cam

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
        parts = load_pickle(osp.join(base_dir, "partitions.pkl"))

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
        return img, self.pids[idx], self.cams[idx], name, self.marks[idx]
