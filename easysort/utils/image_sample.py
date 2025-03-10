from dataclasses import dataclass, asdict
from PIL import Image
from easysort.utils.detections import Detection
import io
import json
from typing import Dict, Any, Union, List, Sequence

@dataclass
class DetectionMetadata:
    uuid: str
    date: str
    robot_id: str

class DetectionSample:
    def __init__(self, images: Sequence[Image.Image], detections: Sequence[Sequence[Detection]], metadata: DetectionMetadata):
        if len(images) != len(detections): raise ValueError("Number of images must match number of detection lists")
        self.images = list(images)
        self.detections = list(detections)
        self.metadata = metadata

    def add(self, image: Image.Image, detections: List[Detection]):
        self.images.append(image)
        self.detections.append(detections)

    def update(self, frame_idx: int, image: Image.Image, detections: List[Detection]):
        self.images[frame_idx] = image
        self.detections[frame_idx] = detections

    def to_json(self) -> str:
        image_data = []
        for image in self.images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'PNG')
            img_byte_arr.seek(0)
            image_data.append(img_byte_arr.getvalue().hex())

        return json.dumps({
            "images": image_data,
            "detections": [[det.to_json() for det in img_dets] for img_dets in self.detections],
            "metadata": asdict(self.metadata)
        })

    @staticmethod
    def from_json(json_data: Union[str, Dict[str, Any]]) -> "DetectionSample":
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        images = []
        for image_data in data["images"]:
            bytes_data = bytes.fromhex(image_data)
            io_bytes = io.BytesIO(bytes_data)
            images.append(Image.open(io_bytes))

        detections = [
            [Detection.from_json(det_data) for det_data in img_dets]
            for img_dets in data["detections"]
        ]

        sample = DetectionSample(
            images=images,
            detections=detections,
            metadata=DetectionMetadata(**data["metadata"])
        )
        return sample

    def __repr__(self) -> str:
        detections_per_image = [len(dets) for dets in self.detections]
        return f"DetectionSample(images={len(self.images)}, detections_per_image={detections_per_image}, metadata={self.metadata})"
