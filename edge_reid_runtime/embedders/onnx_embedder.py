from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from edge_reid_runtime.core.interfaces import Embedder

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class OnnxEmbedderConfig:
    model_path: str
    input_size: Tuple[int, int] = (256, 128)  # (H, W)
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD
    bgr: bool = True


class OnnxReidEmbedder(Embedder):
    def __init__(self, cfg: OnnxEmbedderConfig):
        if ort is None:
            raise ImportError("onnxruntime is not installed. Install with: pip install onnxruntime")
        if cv2 is None:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

        self.cfg = cfg
        self._sess = ort.InferenceSession(cfg.model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._embedding_dim = self._infer_dim()

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        batch = self._preprocess(crops)
        out = self._sess.run([self._output_name], {self._input_name: batch})[0]
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)
        return out.astype(np.float32)

    def _preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        h, w = self.cfg.input_size
        processed = []
        for crop in crops:
            img = crop
            if self.cfg.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(self.cfg.mean, dtype=np.float32)) / np.array(self.cfg.std, dtype=np.float32)
            img = np.transpose(img, (2, 0, 1))  # CHW
            processed.append(img)
        return np.stack(processed, axis=0)

    def _infer_dim(self) -> int:
        h, w = self.cfg.input_size
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        out = self._sess.run([self._output_name], {self._input_name: dummy})[0]
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)
        return int(out.shape[1])
