from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from edge_reid_runtime.core.interfaces import Detector, Detection, Frame

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass(frozen=True)
class YoloV8Config:
    model: str = "yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.7
    imgsz: int = 640
    half: bool = False
    max_det: int = 100


class YoloV8PersonDetector(Detector):
    """
    YOLOv8 wrapper that returns ONLY 'person' detections.

    Output format:
      Detection(
        bbox_xyxy=(x1, y1, x2, y2) in pixel coords,
        conf in [0, 1],
        cls=class_id (COCO person=0),
        cls_name="person"
      )
    NMS: uses YOLO's built-in NMS via predict(conf=..., iou=...).
    """

    def __init__(self, device: str, cfg: Optional[YoloV8Config] = None):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Install with: pip install ultralytics")
        self.cfg = cfg or YoloV8Config()
        self.device = device
        self.model = YOLO(self.cfg.model)

        names: Dict[int, str] = (
            getattr(self.model.model, "names", None)
            or getattr(self.model, "names", {})
            or {}
        )
        self._names = names
        self._person_id = self._find_person_class_id(names)

    @staticmethod
    def _find_person_class_id(names: Dict[int, str]) -> int:
        for k, v in names.items():
            if str(v).lower() == "person":
                return int(k)
        return 0

    def cls_name(self, cls_id: int) -> str:
        return str(self._names.get(int(cls_id), str(cls_id)))

    def detect(self, frame: Frame) -> List[Detection]:
        if frame.image is None:
            return []

        results = self.model.predict(
            source=frame.image,
            device=self.device,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            half=self.cfg.half,
            max_det=self.cfg.max_det,
            classes=[self._person_id],
            verbose=False,
        )
        if not results:
            return []

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        dets: List[Detection] = []
        xyxy = r0.boxes.xyxy
        conf = r0.boxes.conf
        cls = r0.boxes.cls

        for i in range(len(r0.boxes)):
            x1 = float(xyxy[i][0])
            y1 = float(xyxy[i][1])
            x2 = float(xyxy[i][2])
            y2 = float(xyxy[i][3])
            c = float(conf[i])
            k = int(cls[i])
            dets.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=c,
                    cls=k,
                    cls_name=self._names.get(k, "person"),
                )
            )
        return dets
