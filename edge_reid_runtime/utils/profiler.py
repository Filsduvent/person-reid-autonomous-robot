from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

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
    stages_ms: Dict[str, float]


class StageTimer:
    """
    Lightweight context manager for timing named stages.
    Usage:
        with profiler.stage("detector"):
            ...
    """
    def __init__(self, profiler: "StageProfiler", name: str):
        self.profiler = profiler
        self.name = name
        self._t0: Optional[float] = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is None:
            return
        dt = time.perf_counter() - self._t0
        self.profiler._stages_ms[self.name] = self.profiler._stages_ms.get(self.name, 0.0) + dt * 1000.0


class StageProfiler:
    """
    Per-frame profiler with:
    - named stages (ms)
    - total per-frame time (ms)
    - rolling FPS
    - optional RSS via psutil
    """
    def __init__(self, fps_window: int = 30):
        self._t_frame0: Optional[float] = None
        self._dt_window: Deque[float] = deque(maxlen=fps_window)
        self._stages_ms: Dict[str, float] = {}
        self._proc = psutil.Process() if psutil is not None else None

    def stage(self, name: str) -> StageTimer:
        return StageTimer(self, name)

    def on_frame_start(self) -> None:
        self._t_frame0 = time.perf_counter()
        self._stages_ms = {}

    def on_frame_end(self, frame_id: int, timestamp_s: float) -> FrameStats:
        if self._t_frame0 is None:
            self._t_frame0 = time.perf_counter()

        dt = time.perf_counter() - self._t_frame0
        self._dt_window.append(dt)
        dt_ms = dt * 1000.0

        mean_dt = sum(self._dt_window) / max(1, len(self._dt_window))
        fps = (1.0 / mean_dt) if mean_dt > 0 else 0.0

        rss_mb = None
        if self._proc is not None:
            try:
                rss_mb = self._proc.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                rss_mb = None

        if "total" not in self._stages_ms:
            self._stages_ms["total"] = dt_ms

        return FrameStats(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            dt_ms=dt_ms,
            fps_rolling=fps,
            rss_mb=rss_mb,
            stages_ms=dict(self._stages_ms),
        )
