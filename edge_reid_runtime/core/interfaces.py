from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Frame:
    """A single frame unit passed through the pipeline."""
    frame_id: int
    timestamp_s: float
    image: Any  # Typically a numpy array (H,W,3 BGR) from OpenCV, but kept generic
    meta: Dict[str, Any]


@dataclass(frozen=True)
class Detection:
    """Person detection output (bbox in xyxy pixel coords, conf [0,1], cls id/name)."""
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int = 0
    cls_name: Optional[str] = None


@dataclass(frozen=True)
class Track:
    """Tracker output (placeholder for later phases)."""
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float


class InputSource(ABC):
    """Unified frame iterator interface for webcam/video/robot streams."""

    @abstractmethod
    def __iter__(self) -> Iterable[Frame]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        raise NotImplementedError


class Tracker(ABC):
    @abstractmethod
    def update(self, frame: Frame, detections: List[Detection]) -> List[Track]:
        raise NotImplementedError


class Embedder(ABC):
    @abstractmethod
    def embed(self, crops: List[Any]) -> Any:
        """Return embeddings for a list of crops (shape: N x D)."""
        raise NotImplementedError


class GalleryManager(ABC):
    @abstractmethod
    def add(self, identity: str, embedding: Any, meta: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, embedding: Any, topk: int = 5) -> List[Tuple[str, float]]:
        raise NotImplementedError


class IdentityAssigner(ABC):
    @abstractmethod
    def assign(self, track: Track, embedding: Any) -> Tuple[str, float]:
        """Return (identity_label, score)."""
        raise NotImplementedError


class Visualizer(ABC):
    @abstractmethod
    def render(self, frame: Frame, overlays: Dict[str, Any]) -> Any:
        raise NotImplementedError


class Profiler(ABC):
    @abstractmethod
    def on_frame_start(self, frame: Frame) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_frame_end(self, frame: Frame) -> Dict[str, Any]:
        """Return metrics dict for the frame (timings, fps, mem, etc)."""
        raise NotImplementedError
