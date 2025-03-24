from dataclasses import dataclass, asdict
from PIL import Image
from easysort.utils.detections import Detection
import io
import json
from typing import Dict, Any, Union, List, Optional
from dataclasses import field
from uuid import uuid4

@dataclass
class VideoMetadata:
    date: str
    robot_id: str
    uuid: Optional[str] = field(default_factory=lambda: str(uuid4()))

@dataclass
class ImageMetadata:
    frame_idx: int
    timestamp: float
    uuid: str

@dataclass
class ImageSample:
    image: Image.Image
    detections: List[Detection]
    metadata: ImageMetadata

    def to_json(self) -> str:
        img_byte_arr = io.BytesIO()
        self.image.save(img_byte_arr, format=self.image.format or 'PNG')
        img_byte_arr.seek(0)
        image_data = img_byte_arr.getvalue().hex()
        return json.dumps({
            "image": image_data,
            "detections": [det.to_json() for det in self.detections],
            "metadata": asdict(self.metadata)
        })

    @classmethod
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "ImageSample":
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        image_data = bytes.fromhex(data["image"])
        image = Image.open(io.BytesIO(image_data))
        detections = [Detection.from_json(det_data) for det_data in data["detections"]]
        return cls(image, detections, ImageMetadata(**data["metadata"]))


class VideoSample:
    def __init__(self, samples: List[ImageSample], metadata: VideoMetadata):
        self.samples = {sample.metadata.frame_idx: sample for sample in samples}
        self.metadata = metadata

    def update(self, sample: ImageSample):
        self.samples[sample.metadata.frame_idx] = sample

    def to_json(self) -> str:
        return json.dumps({
            "images": [sample.to_json() for sample in self.samples.values()],
            "metadata": asdict(self.metadata)
        })

    @classmethod
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "VideoSample":
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        samples = [ImageSample.from_json(sample) for sample in data["images"]]
        return cls(samples=samples, metadata=VideoMetadata(**data["metadata"]))

    def __repr__(self) -> str:
        return f"VideoSample(samples={len(self.samples)}, detections={[len(s.detections) for s in self.samples]}, metadata={self.metadata})"
