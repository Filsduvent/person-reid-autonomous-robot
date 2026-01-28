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


def validate_run_config(cfg: RunConfig) -> None:
    if cfg.source == "video" and cfg.video_path is None:
        raise ValueError("video_path must be set when source='video'")
    if cfg.source != "webcam" and cfg.webcam_index != 0:
        raise ValueError("webcam_index is only applicable when source='webcam'")
    if cfg.max_frames < 0:
        raise ValueError("max_frames must be >= 0 (0 means unlimited)")
    if cfg.print_every <= 0:
        raise ValueError("print_every must be > 0")
