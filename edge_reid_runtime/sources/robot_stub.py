from __future__ import annotations

import time
from typing import Any, Dict, Iterator

from edge_reid_runtime.core.interfaces import Frame
from edge_reid_runtime.sources.base import BaseSource


class RobotStubSource(BaseSource):
    """
    Placeholder for a real robot stream (ROS2, ZMQ, gRPC, RTSP, etc).
    For now, yields blank frames or raises NotImplemented if you prefer.
    """
    def __init__(self, max_frames: int = 300, width: int = 640, height: int = 480):
        self.max_frames = max_frames
        self.width = width
        self.height = height
        self._closed = False

    def __iter__(self) -> Iterator[Frame]:
        # No numpy dependency here; image is None by design (pipeline currently does no vision ops)
        for frame_id in range(self.max_frames):
            if self._closed:
                break
            ts = time.time()
            meta: Dict[str, Any] = {"source": "robot_stub"}
            yield Frame(frame_id=frame_id, timestamp_s=ts, image=None, meta=meta)
            time.sleep(1.0 / 30.0)  # simulate ~30 FPS stream

    def close(self) -> None:
        self._closed = True
