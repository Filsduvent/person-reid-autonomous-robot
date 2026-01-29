from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from edge_reid_runtime.embedders.torch_embedder import TorchEmbedderConfig, TorchReidEmbedder


SUPPORTED_BACKBONES = (
    "osnet_x0_25",
    "osnet_x0_5",
    "mobilenetv3_small",
    "mobilenetv3_large",
    "shufflenetv2_x0_5",
    "shufflenetv2_x1_0",
    "efficientnet_lite0",
    "efficientnet_lite1",
)


@dataclass(frozen=True)
class EmbedderConfig:
    backbone: str
    device: str = "cpu"
    weights: Optional[str] = None
    input_size: Optional[Tuple[int, int]] = None  # (H, W)


def _default_input_size(backbone: str) -> Tuple[int, int]:
    name = backbone.lower()
    if name.startswith("osnet"):
        return (256, 128)
    return (224, 224)


def create_embedder(cfg: EmbedderConfig) -> TorchReidEmbedder:
    name = cfg.backbone.lower()
    if name == "mobilenetv3":
        name = "mobilenetv3_small"
    if name == "shufflenetv2":
        name = "shufflenetv2_x1_0"
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{cfg.backbone}'. Supported: {SUPPORTED_BACKBONES}")

    input_size = cfg.input_size or _default_input_size(name)
    tcfg = TorchEmbedderConfig(
        backbone=name,
        device=cfg.device,
        weights=cfg.weights,
        input_size=input_size,
    )
    return TorchReidEmbedder(tcfg)
