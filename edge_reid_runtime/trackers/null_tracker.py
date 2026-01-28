from __future__ import annotations

from typing import List

from edge_reid_runtime.core.interfaces import Tracker, Track, Frame, Detection


class NullTracker(Tracker):
    """
    Deterministic stub tracker.

    - If there is 1 detection, it always returns track_id=1
    - If multiple detections, assigns track_ids 1..N deterministically by detection index
    """

    def __init__(self):
        self._next_id = 1  # kept for future extension; currently deterministic

    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        tracks: List[Track] = []
        for i, det in enumerate(detections):
            track_id = i + 1
            tracks.append(Track(track_id=track_id, bbox_xyxy=det.bbox_xyxy, conf=det.conf))
        return tracks
