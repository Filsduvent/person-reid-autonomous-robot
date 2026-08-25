from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge_reid_runtime.detectors.null_detector import NullDetector
from edge_reid_runtime.detectors.yolo_v8 import YoloV8Config, YoloV8PersonDetector


@dataclass(frozen=True)
class DetectorConfig:
    family: str = "yolo26"
    weights: str = "edge_reid_runtime/weights/yolo26/yolo26n.pt"
    conf: float = 0.35
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 100
    half: bool = False


def build_detector(cfg: DetectorConfig, device: str):
    """Create the configured detector without coupling the rest of the pipeline to YOLO."""
    family = cfg.family.lower()
    if family == "null":
        return NullDetector(mode="empty")
    if family not in {"yolo26", "yolov8"}:
        raise ValueError(f"Unsupported detector family '{cfg.family}'.")
    if family == "yolo26":
        path = Path(cfg.weights)
        if path.name != "yolo26n.pt":
            raise ValueError("Qualification YOLO26 requires the official yolo26n.pt checkpoint.")
        if not path.is_file():
            raise FileNotFoundError(f"Official YOLO26 checkpoint is required at: {path}")
    return YoloV8PersonDetector(
        device=device,
        cfg=YoloV8Config(
            model=cfg.weights, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz,
            max_det=cfg.max_det, half=cfg.half, end2end=(family == "yolo26"),
        ),
    )
