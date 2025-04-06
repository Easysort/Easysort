import numpy as np
from ultralytics.engine.results import Results
from typing import List, Optional, Tuple, Dict, Union, Any
import cv2

class Detection:
    def __init__(self, box: np.ndarray, mask: Optional[np.ndarray] = None, class_id: int = -1, confidence: Optional[float] = None,
                 names: Optional[Dict[Union[int, str], str]] = None, timestamp: Optional[float] = None) -> None:
        self.box = box
        self.mask = mask
        self.class_id = class_id
        self.confidence = confidence
        self.names = names if names is not None else {}
        self.class_name = self.names.get(str(class_id), "")
        self.xyxy = [int(x) for x in self.box]
        self.area = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])
        self._center_point: Optional[Tuple[float, float, float]] = None
        self.timestamp = timestamp
        self._robot_center_point: Optional[np.ndarray] = None

    def set_center_point(self, center_point: Optional[Tuple[float, float, float]] = None) -> None:
        self._center_point = center_point

    @property
    def center_point(self) -> Tuple[float, float, float]:
        if self._center_point is not None: return self._center_point
        if self.mask is None: center_point = ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2, 0)
        else:
            center_point = np.mean(np.argwhere(self.mask), axis=0)
            center_point = (center_point[1], center_point[0], 0) # for some reason this is reversed
        self._center_point = center_point
        return center_point

    def current_center_point(self, speed: float) -> Tuple[float, float, float]:
        if self.timestamp is None: raise NotImplementedError("Timestamp is not set")
        return (self.center_point[0] + speed * self.timestamp, self.center_point[1], self.center_point[2])

    def to_json(self):
        return {
            "box": self.box.tolist(),
            "mask": self.mask.tolist() if self.mask is not None else None,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "names": self.names,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Detection":
        names = {int(k) if k.isdigit() else k: v for k, v in data.get("names", {}).items()}
        return cls(
            box=np.array(data["box"]),
            mask=np.array(data["mask"]) if data.get("mask") is not None else None,
            class_id=data["class_id"],
            confidence=data["confidence"],
            names=names,
            timestamp=data.get("timestamp")
        )

    def __repr__(self):
        return f"Detection(box={self.box}, mask={self.mask}, class_name={self.class_name}, confidence={self.confidence})"

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

class Detections:
    @staticmethod
    def from_ultralytics(result: Results) -> List[Detection]:
        return [
            Detection(box=box.cpu().numpy(), mask=None, class_id=int(class_id.cpu().numpy()), confidence=confidence.cpu().numpy(), names=result.names)
            for box, class_id, confidence in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)
        ]