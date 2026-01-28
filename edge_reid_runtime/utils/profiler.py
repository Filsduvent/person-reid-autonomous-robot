from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

try:
    import psutil  # optional
except Exception:  # pragma: no cover
    psutil = None


@dataclass
class FrameStats:
    frame_id: int
    timestamp_s: float
    dt_ms: float
    fps_rolling: float
    rss_mb: Optional[float]


class SimpleProfiler:
    """
    Minimal real-time profiler:
    - measures per-frame total loop time (ms)
    - rolling FPS over a window
    - optional RSS memory (psutil)
    """
    def __init__(self, fps_window: int = 30, enable_memory: bool = True):
        self._t0: Optional[float] = None
        self._dt_window: Deque[float] = deque(maxlen=fps_window)
        self._enable_memory = enable_memory
        self._proc = psutil.Process() if (psutil is not None and enable_memory) else None

    def on_frame_start(self) -> None:
        self._t0 = time.perf_counter()

    def on_frame_end(
        self, frame_id: int, timestamp_s: Optional[float] = None
    ) -> FrameStats:
        if self._t0 is None:
            # Keep robust, but report 0-duration for missing start.
            self._t0 = time.perf_counter()
            dt = 0.0
        else:
            dt = time.perf_counter() - self._t0
        dt_ms = dt * 1000.0
        self._dt_window.append(dt)

        # Rolling FPS: 1 / mean(dt)
        mean_dt = sum(self._dt_window) / max(1, len(self._dt_window))
        fps = (1.0 / mean_dt) if mean_dt > 0 else 0.0

        rss_mb = None
        if self._proc is not None:
            try:
                rss_mb = self._proc.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                rss_mb = None

        return FrameStats(
            frame_id=frame_id,
            timestamp_s=timestamp_s if timestamp_s is not None else time.time(),
            dt_ms=dt_ms,
            fps_rolling=fps,
            rss_mb=rss_mb,
        )
