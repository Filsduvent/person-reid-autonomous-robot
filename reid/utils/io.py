import os
import os.path as osp
import pickle

def expand(path: str) -> str:
    return osp.abspath(osp.expanduser(path))

def load_pickle(path: str):
    path = expand(path)
    if not osp.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
