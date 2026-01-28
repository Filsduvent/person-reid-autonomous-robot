from __future__ import annotations

from pathlib import Path

from edge_reid_runtime.core.types import RunConfig
from edge_reid_runtime.sources.webcam import WebcamSource
from edge_reid_runtime.sources.video import VideoFileSource
from edge_reid_runtime.sources.robot_stub import RobotStubSource
from edge_reid_runtime.sources.base import BaseSource


def create_source(cfg: RunConfig) -> BaseSource:
    if cfg.source == "webcam":
        return WebcamSource(index=cfg.webcam_index, max_frames=cfg.max_frames)
    if cfg.source == "video":
        if cfg.video_path is None:
            raise ValueError("video_path is required when --source=video")
        return VideoFileSource(path=cfg.video_path, max_frames=cfg.max_frames)
    if cfg.source == "robot":
        return RobotStubSource(max_frames=(cfg.max_frames if cfg.max_frames > 0 else 300))
    raise ValueError(f"Unknown source: {cfg.source}")
