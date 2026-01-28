from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class RunConfig:
    source: Literal["webcam", "video", "robot"]
    device: Literal["cpu", "cuda", "auto"]
    output_dir: Path
    video_path: Optional[Union[Path, str]] = None
    webcam_index: int = 0
    max_frames: int = 0         # 0 => unlimited
    print_every: int = 10
    reid_backbone: Optional[str] = None
    weights: Optional[Union[Path, str]] = None
    detector: str = "yolov8"
    yolo_model: str = "yolov8n.pt"
    det_conf: float = 0.35
    det_iou: float = 0.70
    imgsz: int = 640
    max_det: int = 100
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    save_video: bool = False
    display: bool = False
    output_video: Optional[Path] = None


def validate_run_config(cfg: RunConfig) -> None:
    if cfg.source == "video" and cfg.video_path is None:
        raise ValueError("video_path must be set when source='video'")
    if cfg.source != "webcam" and cfg.webcam_index != 0:
        raise ValueError("webcam_index is only applicable when source='webcam'")
    if cfg.max_frames < 0:
        raise ValueError("max_frames must be >= 0 (0 means unlimited)")
    if cfg.print_every <= 0:
        raise ValueError("print_every must be > 0")
    if cfg.max_age < 0:
        raise ValueError("max_age must be >= 0")
    if cfg.n_init < 1:
        raise ValueError("n_init must be >= 1")
    if cfg.max_iou_distance < 0:
        raise ValueError("max_iou_distance must be >= 0")
