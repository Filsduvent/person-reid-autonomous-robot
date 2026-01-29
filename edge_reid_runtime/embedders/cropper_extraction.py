from __future__ import annotations

from typing import List, Tuple

import numpy as np

from edge_reid_runtime.core.interfaces import Track


def _clip_bbox(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def extract_track_crops(
    image: np.ndarray,
    tracks: List[Track],
    min_size: int = 10,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract person crops for active tracks.
    Returns: (crops, track_ids)
    """
    crops: List[np.ndarray] = []
    track_ids: List[int] = []

    if image is None:
        return crops, track_ids

    height, width = image.shape[:2]
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr.bbox_xyxy)
        x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, width, height)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        track_ids.append(int(tr.track_id))

    return crops, track_ids
