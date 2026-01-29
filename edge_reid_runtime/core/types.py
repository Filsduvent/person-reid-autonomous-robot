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
    known_threshold: float = 0.55
    unknown_threshold: float = 0.45
    max_identities: int = 500
    id_prefix: str = "person_"
    id_width: int = 4
    update_threshold: float = 0.65
    margin_threshold: float = 0.15
    stable_age: int = 10
    stable_hits: int = 5
    min_det_conf: float = 0.35
    area_drop_ratio: float = 0.5
    aspect_ratio_min: float = 0.2
    aspect_ratio_max: float = 0.9
    ema_alpha: float = 0.1
    reacquire_cooldown_frames: int = 15


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
    if not (0.0 <= cfg.known_threshold <= 1.0):
        raise ValueError("known_threshold must be in [0,1]")
    if not (0.0 <= cfg.unknown_threshold <= 1.0):
        raise ValueError("unknown_threshold must be in [0,1]")
    if cfg.unknown_threshold >= cfg.known_threshold:
        raise ValueError("unknown_threshold must be < known_threshold")
    if not (0.0 <= cfg.update_threshold <= 1.0):
        raise ValueError("update_threshold must be in [0,1]")
    if cfg.margin_threshold < 0:
        raise ValueError("margin_threshold must be >= 0")
    if cfg.stable_age < 0:
        raise ValueError("stable_age must be >= 0")
    if cfg.stable_hits < 0:
        raise ValueError("stable_hits must be >= 0")
    if not (0.0 <= cfg.min_det_conf <= 1.0):
        raise ValueError("min_det_conf must be in [0,1]")
    if not (0.0 < cfg.area_drop_ratio <= 1.0):
        raise ValueError("area_drop_ratio must be in (0,1]")
    if cfg.aspect_ratio_min <= 0 or cfg.aspect_ratio_max <= 0:
        raise ValueError("aspect_ratio_min/max must be > 0")
    if cfg.aspect_ratio_min >= cfg.aspect_ratio_max:
        raise ValueError("aspect_ratio_min must be < aspect_ratio_max")
    if not (0.0 < cfg.ema_alpha <= 1.0):
        raise ValueError("ema_alpha must be in (0,1]")
    if cfg.max_identities < 1:
        raise ValueError("max_identities must be >= 1")
    if cfg.id_width < 1:
        raise ValueError("id_width must be >= 1")
    if cfg.reacquire_cooldown_frames < 0:
        raise ValueError("reacquire_cooldown_frames must be >= 0")
