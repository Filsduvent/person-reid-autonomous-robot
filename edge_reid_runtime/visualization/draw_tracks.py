from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

from edge_reid_runtime.core.interfaces import Track

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


_TRACK_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 127, 0),
    (127, 255, 0),
    (0, 127, 255),
    (127, 0, 255),
)


_TRACK_TRAILS: Dict[int, Deque[Tuple[int, int]]] = {}


def _color_for_track(track_id: int) -> Tuple[int, int, int]:
    return _TRACK_PALETTE[int(track_id) % len(_TRACK_PALETTE)]


def _text_color_for_bg(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bgr
    # Simple luminance to pick white/black for contrast.
    luminance = (0.114 * b) + (0.587 * g) + (0.299 * r)
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)


def draw_tracks(
    image,
    tracks: List[Track],
    show_trajectory: bool = False,
    trajectory_length: int = 30,
) -> None:
    if cv2 is None:
        raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

    active_ids = {int(tr.track_id) for tr in tracks}
    if show_trajectory:
        # Prune trails for inactive IDs to avoid unbounded growth.
        for tid in list(_TRACK_TRAILS.keys()):
            if tid not in active_ids:
                del _TRACK_TRAILS[tid]

    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.bbox_xyxy)
        color = _color_for_track(tr.track_id)
        text_color = _text_color_for_bg(color)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        age = None
        if tr.meta and "age" in tr.meta:
            try:
                age = int(tr.meta["age"])
            except Exception:
                age = None
        label = f"ID {tr.track_id} | {tr.conf:.2f}"
        if age is not None:
            label += f" | age {age}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pad = 4
        box_top = y1 - th - (2 * pad)
        box_bottom = y1
        text_y = y1 - pad
        if box_top < 0:
            box_top = y1
            box_bottom = y1 + th + (2 * pad)
            text_y = y1 + th + pad
        cv2.rectangle(image, (x1, box_top), (x1 + tw + (2 * pad), box_bottom), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + pad, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
        )

        if show_trajectory:
            tid = int(tr.track_id)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            trail = _TRACK_TRAILS.setdefault(tid, deque(maxlen=max(1, int(trajectory_length))))
            trail.append((cx, cy))
            if len(trail) >= 2:
                pts = list(trail)
                for i in range(1, len(pts)):
                    cv2.line(image, pts[i - 1], pts[i], color, 2)
