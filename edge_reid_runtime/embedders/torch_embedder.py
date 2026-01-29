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
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TorchEmbedderConfig:
    backbone: str
    device: str = "cpu"
    weights: Optional[str] = None
    input_size: Tuple[int, int] = (256, 128)  # (H, W) default for reid
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD
    bgr: bool = True


class TorchReidEmbedder(Embedder):
    """
    Torch-based embedder wrapper.
    - Supports torchreid OSNet backbones.
    - Supports torchvision MobileNetV3/ShuffleNetV2.
    - EfficientNet-Lite requires timm (optional).
    """

    def __init__(self, cfg: TorchEmbedderConfig):
        if torch is None or nn is None:
            raise ImportError("torch is not installed. Install with: pip install torch torchvision")
        if cv2 is None:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

        self.cfg = cfg
        self.device = cfg.device
        self.model = self._build_model(cfg.backbone, cfg.weights)
        self.model.eval().to(self.device)
        self._embedding_dim = self._infer_dim()

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        batch = self._preprocess(crops)
        with torch.no_grad():
            feats = self.model(batch)

        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return feats.detach().cpu().numpy().astype(np.float32)

    def _preprocess(self, crops: List[np.ndarray]) -> "torch.Tensor":
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
        batch = np.stack(processed, axis=0)
        return torch.from_numpy(batch).to(self.device)

    def _infer_dim(self) -> int:
        h, w = self.cfg.input_size
        dummy = torch.zeros((1, 3, h, w), device=self.device)
        with torch.no_grad():
            out = self.model(dummy)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        return int(out.shape[1])

    def _build_model(self, backbone: str, weights: Optional[str]):
        name = backbone.lower()

        if name.startswith("osnet"):
            try:
                import torchreid
            except Exception as exc:  # pragma: no cover
                raise ImportError("torchreid is required for OSNet backbones.") from exc

            model = torchreid.models.build_model(
                name=name,
                num_classes=1,
                loss="softmax",
                pretrained=False,
            )
            if weights:
                try:
                    torchreid.utils.load_pretrained_weights(model, weights)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load weights from {weights}: {exc}") from exc
            return model

        if name in ("mobilenetv3_small", "mobilenetv3_large"):
            from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

            model = mobilenet_v3_small(weights=None) if name.endswith("small") else mobilenet_v3_large(weights=None)
            model.classifier = nn.Identity()
            if weights:
                self._load_state_dict(model, weights)
            return model

        if name in ("shufflenetv2_x0_5", "shufflenetv2_x1_0"):
            from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0

            model = shufflenet_v2_x0_5(weights=None) if name.endswith("x0_5") else shufflenet_v2_x1_0(weights=None)
            model.fc = nn.Identity()
            if weights:
                self._load_state_dict(model, weights)
            return model

        if name.startswith("efficientnet_lite"):
            try:
                import timm
            except Exception as exc:  # pragma: no cover
                raise ImportError("timm is required for EfficientNet-Lite backbones.") from exc

            model = timm.create_model(name, pretrained=False, num_classes=0)
            if weights:
                self._load_state_dict(model, weights)
            return model

        raise ValueError(f"Unsupported backbone '{backbone}'.")

    @staticmethod
    def _load_state_dict(model, weights_path: str) -> None:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
