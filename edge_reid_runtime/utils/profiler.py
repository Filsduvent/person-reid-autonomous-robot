from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

try:
    import psutil  # optional
except Exception:  # pragma: no cover
    psutil = None
try:
    import torch  # optional
except Exception:  # pragma: no cover
    torch = None


@dataclass
class FrameStats:
    frame_id: int
    timestamp_s: float
    dt_ms: float
    fps_rolling: float
    rss_mb: Optional[float]
    vram_mb: Optional[float]
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
        self.profiler._maybe_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is None:
            return
        self.profiler._maybe_sync()
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
    def __init__(self, fps_window: int = 30, collect_history: bool = False, cuda_sync: bool = False):
        self._t_frame0: Optional[float] = None
        self._dt_window: Deque[float] = deque(maxlen=fps_window)
        self._stages_ms: Dict[str, float] = {}
        self._proc = psutil.Process() if psutil is not None else None
        self._collect_history = collect_history
        self._history: List[FrameStats] = []
        self._cuda_available = bool(torch is not None and torch.cuda.is_available())
        self._cuda_sync = bool(cuda_sync) and self._cuda_available

    def stage(self, name: str) -> StageTimer:
        return StageTimer(self, name)

    def _maybe_sync(self) -> None:
        if self._cuda_sync and torch is not None:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

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
        vram_mb = None
        if self._cuda_available:
            try:
                vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            except Exception:
                vram_mb = None

        if "total" not in self._stages_ms:
            self._stages_ms["total"] = dt_ms

        stats = FrameStats(
            frame_id=frame_id,
            timestamp_s=timestamp_s,
            dt_ms=dt_ms,
            fps_rolling=fps,
            rss_mb=rss_mb,
            vram_mb=vram_mb,
            stages_ms=dict(self._stages_ms),
        )
        if self._collect_history:
            self._history.append(stats)
        return stats

    @staticmethod
    def _summarize(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0}
        vals = sorted(values)
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        p95_idx = int(0.95 * (n - 1))
        p95 = vals[p95_idx]
        return {"mean": mean, "median": median, "p95": p95}

    def measured_history(self, warmup_frames: int = 0) -> List[FrameStats]:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be >= 0")
        return self._history[warmup_frames:]

    def summarize(self, warmup_frames: int = 0) -> Dict[str, Dict[str, float]]:
        history = self.measured_history(warmup_frames)
        if not history:
            return {}
        totals = [h.dt_ms for h in history]
        out: Dict[str, Dict[str, float]] = {"total": self._summarize(totals)}
        stage_keys = set()
        for h in history:
            stage_keys.update(h.stages_ms.keys())
        for k in sorted(stage_keys):
            out[k] = self._summarize([h.stages_ms.get(k, 0.0) for h in history])
        return out

    def rss_summary(self, warmup_frames: int = 0) -> Dict[str, Optional[float]]:
        values = [h.rss_mb for h in self.measured_history(warmup_frames) if h.rss_mb is not None]
        if not values:
            return {"rss_mean_mb": None, "rss_peak_mb": None}
        return {"rss_mean_mb": sum(values) / len(values), "rss_peak_mb": max(values)}
