from __future__ import annotations

from typing import List

from edge_reid_runtime.core.interfaces import Detector, Detection, Frame


class NullDetector(Detector):
    """
    Deterministic stub detector.

    Modes:
    - "empty": returns no detections
    - "center": returns a single centered bbox (useful to exercise tracking paths)
    """

    def __init__(self, mode: str = "empty", conf: float = 0.99):
        if mode not in ("empty", "center"):
            raise ValueError(f"NullDetector mode must be 'empty' or 'center', got: {mode}")
        self.mode = mode
        self.conf = float(conf)

    def detect(self, frame: Frame) -> List[Detection]:
        if self.mode == "empty":
            return []

        # If image is available, make a centered bbox. If image is None (robot stub), fallback bbox.
        if frame.image is not None:
            try:
                h, w = frame.image.shape[:2]
            except Exception:
                h, w = 480, 640
        else:
            h, w = 480, 640

        # Center bbox covering ~40% width, ~70% height (typical person crop region)
        bw = w * 0.4
        bh = h * 0.7
        x1 = (w - bw) / 2.0
        y1 = (h - bh) / 2.0
        x2 = x1 + bw
        y2 = y1 + bh

        return [Detection(bbox_xyxy=(x1, y1, x2, y2), conf=self.conf, cls=0)]
