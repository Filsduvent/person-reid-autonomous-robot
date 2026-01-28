from __future__ import annotations

from pathlib import Path

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class VideoWriter:
    """
    Lazy-initialized video writer. Creates the file on first frame.
    """
    def __init__(self, out_path: Path, fps: float = 30.0):
        if cv2 is None:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
        self.out_path = Path(out_path)
        self.fps = float(fps)
        self._writer = None

    def write(self, frame_bgr) -> None:
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter at: {self.out_path}")
        self._writer.write(frame_bgr)

    def close(self) -> None:
        try:
            if self._writer is not None:
                self._writer.release()
        except Exception:
            pass
