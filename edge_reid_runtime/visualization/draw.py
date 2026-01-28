from __future__ import annotations

from typing import Callable, List, Optional

from edge_reid_runtime.core.interfaces import Detection

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def draw_detections(
    image,
    detections: List[Detection],
    cls_name_fn: Optional[Callable[[int], str]] = None,
) -> None:
    """
    In-place drawing. Requires OpenCV.
    - image: numpy array (BGR)
    - cls_name_fn: optional function int->str
    """
    if cv2 is None:
        raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        det_color = (255, 255, 0)  # cyan for detections
        cv2.rectangle(image, (x1, y1), (x2, y2), det_color, 2)

        name = "person" if cls_name_fn is None else cls_name_fn(d.cls)
        label = f"{name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 6, y1), det_color, -1)
        cv2.putText(image, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
