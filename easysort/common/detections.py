import numpy as np
from ultralytics.engine.results import Results
from typing import List, Optional, Tuple, Dict
import cv2

class Detection:
    def __init__(self, box: np.ndarray, mask: Optional[np.ndarray] = None, class_id: int = -1,
                 conf: Optional[float] = None, names: Optional[Dict[int, str]] = None) -> None:
        self.box = box
        self.mask = mask
        self.class_id = class_id
        self.confidence = conf
        self.names = names if names is not None else {}
        self.class_name = names[class_id] if names is not None else ""
        self.xyxy = [int(x) for x in self.box]
        self.area = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
        self._center_point: Optional[Tuple[float, float]] = None

    @staticmethod
    def from_ultralytics(result: Results) -> List["Detection"]:
        return [
            Detection(box=box.cpu().numpy(), mask=None, class_id=int(class_id.cpu().numpy()), conf=confidence.cpu().numpy(), names=result.names)
            for box, class_id, confidence in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)
        ]

    @property
    def center_point(self) -> Tuple[float, float]:
        if self._center_point is not None:
            return self._center_point
        if self.mask is None:
            center_point = ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)
        else:
            center_point = np.mean(np.argwhere(self.mask), axis=0)
            center_point = (center_point[1], center_point[0]) # for some reason this is reversed
        self._center_point = center_point
        return center_point

class Mask(np.ndarray):
    def __new__(cls, mask: np.ndarray):
        obj = np.asarray(mask).view(cls)
        return obj

    @staticmethod
    def from_ultralytics(result: Results, image_shape: tuple) -> List[np.ndarray]:
        assert len(result) == 1, "Only one result is supported"
        return [
            cv2.resize(mask.cpu().numpy(), (image_shape[1], image_shape[0])) for mask in result[0].masks.data
        ]
