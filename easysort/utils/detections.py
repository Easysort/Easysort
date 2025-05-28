from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics.engine.results import Results


@dataclass
class Detection:
    box: np.ndarray
    mask: np.ndarray | None = None
    class_id: int = -1
    confidence: float | None = None
    names: dict[int | str, str] | None = field(default_factory=dict)
    timestamp: float | None = None
    _center_point: tuple[float, float, float] | None = field(default=None, init=False, repr=False)
    _robot_center_point: np.ndarray | None = field(default=None, init=False, repr=False)

    def __eq__(self, other):
        if not isinstance(other, Detection):
            return NotImplemented
        return (
            np.array_equal(self.box, other.box)
            and (
                np.array_equal(self.mask, other.mask)
                if self.mask is not None and other.mask is not None
                else self.mask == other.mask
            )
            and self.class_id == other.class_id
            and self.confidence == other.confidence
            and self.names == other.names
            and self.timestamp == other.timestamp
        )

    def __post_init__(self):
        if self.names is None:
            self.names = {}
        self.class_name = self.names.get(str(self.class_id), "")
        self.xyxy = [int(x) for x in self.box]
        self.area = (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])

    def set_center_point(self, center_point: tuple[float, float, float] | None = None) -> None:
        self._center_point = center_point

    @property
    def center_point(self) -> tuple[float, float, float]:
        if self._center_point is not None:
            return self._center_point
        if self.mask is None:
            center_point = ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2, 0)
        else:
            center_point = np.mean(np.argwhere(self.mask), axis=0)
            center_point = (center_point[1], center_point[0], 0)  # for some reason this is reversed
        self._center_point = center_point
        return center_point

    def current_center_point(self, speed: float) -> tuple[float, float, float]:
        if self.timestamp is None:
            raise NotImplementedError("Timestamp is not set")
        return (self.center_point[0] + speed * self.timestamp, self.center_point[1], self.center_point[2])

    def to_dict(self):
        return {
            "box": self.box.tolist(),
            "mask": self.mask.tolist() if self.mask is not None else None,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "names": self.names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        names = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in data.get("names", {}).items()}
        return cls(
            box=np.array(data["box"]),
            mask=np.array(data["mask"]) if data.get("mask") is not None else None,
            class_id=data["class_id"],
            confidence=data["confidence"],
            names=names,
            timestamp=data.get("timestamp"),
        )

    def __repr__(self):
        return (
            f"Detection(box={self.box}, mask={self.mask}, class_name={self.class_name}, confidence={self.confidence})"
        )


class Mask(np.ndarray):
    def __new__(cls, mask: np.ndarray):
        obj = np.asarray(mask).view(cls)
        return obj

    @staticmethod
    def from_ultralytics(result: list[Results], image_shape: tuple) -> list[np.ndarray]:
        assert len(result) == 1, "Only one result is supported"
        if not result[0].masks:
            return []
        return [cv2.resize(mask.cpu().numpy(), (image_shape[1], image_shape[0])) for mask in result[0].masks.data]


class Detections:
    @staticmethod
    def from_ultralytics(result: Results) -> list[Detection]:
        if not result.boxes:
            return []
        return [
            Detection(
                box=box.cpu().numpy(),
                mask=None,
                class_id=int(class_id.cpu().numpy()),
                confidence=float(confidence.cpu().numpy()),
                names=result.names,
            )
            for box, class_id, confidence in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)
        ]
