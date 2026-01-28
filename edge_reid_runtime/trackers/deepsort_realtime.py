from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from edge_reid_runtime.core.interfaces import Tracker, Track, Detection, Frame

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # pragma: no cover
    DeepSort = None


@dataclass(frozen=True)
class DeepSortConfig:
    # Lifecycle
    max_age: int = 30          # frames to keep "lost" tracks before deletion
    n_init: int = 3            # hits before confirmed
    max_iou_distance: float = 0.7  # association gate (IoU distance)

    # Preprocessing
    # Important: YOLO already applies NMS. Keep DeepSORT preprocessing NMS disabled.
    nms_max_overlap: float = 1.0   # 1.0 => skip internal NMS in deep-sort-realtime

    # Appearance embedder (used until you plug your own embeddings)
    # Options supported by deep-sort-realtime include 'mobilenet', 'torchreid', 'clip' (see project docs).
    embedder: str = "mobilenet"
    half: bool = False
    bgr: bool = True
    embedder_gpu: bool = False     # keep False for Raspberry Pi target


class DeepSortRealtimeTracker(Tracker):
    """
    DeepSORT tracker wrapper.

    Input detections: List[Detection] with bbox_xyxy
    Internal format expected by deep-sort-realtime:
      ([left, top, w, h], confidence, detection_class)  (per PyPI docs)
    Output tracks: List[Track] with bbox_xyxy (absolute pixels), stable track_id

    Note: deep-sort-realtime's update_tracks API expects ltwh detections and can take a frame
    to compute appearance embeddings.
    """

    def __init__(self, cfg: Optional[DeepSortConfig] = None):
        if DeepSort is None:
            raise ImportError("deep-sort-realtime is not installed. Install with: pip install deep-sort-realtime")
        self.cfg = cfg or DeepSortConfig()

        # We keep kwargs explicit and stable; if you upgrade deep-sort-realtime and something changes,
        # it's easy to adjust here without touching the pipeline.
        self._tracker = DeepSort(
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            max_iou_distance=self.cfg.max_iou_distance,
            nms_max_overlap=self.cfg.nms_max_overlap,
            embedder=self.cfg.embedder,
            half=self.cfg.half,
            bgr=self.cfg.bgr,
            embedder_gpu=self.cfg.embedder_gpu,
        )

    @staticmethod
    def _xyxy_to_ltwh(x1: float, y1: float, x2: float, y2: float) -> List[float]:
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return [float(x1), float(y1), float(w), float(h)]

    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        # Convert to deep-sort-realtime expected format:
        raw = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            raw.append((self._xyxy_to_ltwh(x1, y1, x2, y2), float(d.conf), int(d.cls)))

        # deep-sort-realtime can compute embeddings from the frame if embedder is enabled.
        frame_image = frame.image if frame is not None else None
        tracks = self._tracker.update_tracks(raw, frame=frame_image)

        # Collect confirmed tracks
        tracks_out: List[Track] = []

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = int(t.track_id)

            # Prefer original detection bbox when available:
            ltrb = t.to_ltrb(orig=True)
            if ltrb is None:
                ltrb = t.to_ltrb()
            if ltrb is None:
                continue

            x1, y1, x2, y2 = map(float, ltrb)

            # Best-effort confidence: deep-sort-realtime track may expose det_conf, otherwise None.
            conf_raw = getattr(t, "det_conf", 1.0)
            conf = float(conf_raw) if conf_raw is not None else 1.0

            tracks_out.append(Track(track_id=tid, bbox_xyxy=(x1, y1, x2, y2), conf=conf))

        return tracks_out
