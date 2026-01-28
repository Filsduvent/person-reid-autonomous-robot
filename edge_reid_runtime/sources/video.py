from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, Union
from urllib.parse import urlparse

from edge_reid_runtime.core.interfaces import Frame
from edge_reid_runtime.sources.base import BaseSource

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class VideoFileSource(BaseSource):
    def __init__(self, path: Union[Path, str], max_frames: int = 0):
        if cv2 is None:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
        self.path = Path(path) if not isinstance(path, str) else path
        if not _is_url(self.path):
            path_obj = Path(self.path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Video file not found: {path_obj}")
            self._capture_src = str(path_obj)
        else:
            self._capture_src = str(self.path)
        self.max_frames = max_frames

        self.cap = cv2.VideoCapture(self._capture_src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self._capture_src}")

        self._closed = False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 0 else None

    def __iter__(self) -> Iterator[Frame]:
        frame_id = 0
        while True:
            if self._closed:
                break
            ok, img = self.cap.read()
            if not ok:
                break

            ts = time.time()
            meta: Dict[str, Any] = {"source": "video", "path": str(self._capture_src)}
            if self._fps is not None:
                meta["fps"] = self._fps
            yield Frame(frame_id=frame_id, timestamp_s=ts, image=img, meta=meta)

            frame_id += 1
            if self.max_frames > 0 and frame_id >= self.max_frames:
                break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.cap.release()
        except Exception:
            pass


def _is_url(path: Union[Path, str]) -> bool:
    value = str(path)
    return urlparse(value).scheme in ("http", "https", "rtsp", "rtsps")
