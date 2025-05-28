import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from easysort.utils.detections import Detection
from easysort.utils.lossless_video import read_mkv, save_lossless_mkv


@dataclass
class ImageMetadata:
    frame_idx: int
    timestamp: float
    uuid: str
    detections: list[Detection] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "uuid": self.uuid,
            "detections": [det.to_dict() for det in self.detections] if self.detections else None,
        }

    @classmethod
    def from_dict(cls, data) -> "ImageMetadata":
        detections = data.get("detections")
        if detections is not None:
            detections = [Detection.from_dict(d) for d in detections]
        return cls(
            frame_idx=data["frame_idx"],
            timestamp=data["timestamp"],
            uuid=data["uuid"],
            detections=detections,
        )


@dataclass
class ImageSample:
    image: Image.Image
    metadata: ImageMetadata

    def save_image(self, path: Path) -> None:
        assert path.suffix == ".png", "Image must be saved as PNG"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, format="PNG", optimize=True)

    def save_metadata(self, path: Path) -> None:
        assert path.suffix == ".json", "Metadata must be saved as JSON"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.metadata.to_dict(), f)

    @classmethod
    def load(cls, image_path: Path, metadata_path: Path) -> "ImageSample":
        image = Image.open(image_path)
        with metadata_path.open("r") as f:
            metadata = ImageMetadata.from_dict(json.load(f))
        return cls(image=image, metadata=metadata)


@dataclass
class VideoMetadata:
    date: str
    robot_id: str
    uuid: str = field(default_factory=lambda: str(uuid4()))


class VideoSample:
    def __init__(self, samples: list[ImageSample], metadata: VideoMetadata):
        self.samples = {sample.metadata.frame_idx: sample for sample in samples}
        self.metadata = metadata
        assert all(sample.metadata.uuid == metadata.uuid for sample in samples), (
            "All samples must have the same UUID as the video metadata"
        )

    @property
    def samples_list(self) -> list[ImageSample]:
        return [self.samples[i] for i in sorted(self.samples.keys())]

    def update(self, sample: ImageSample):
        assert sample.metadata.uuid == self.metadata.uuid, "Sample UUID does not match video metadata UUID"
        self.samples[sample.metadata.frame_idx] = sample

    def save_video(self, path: Path, fps: int = 1) -> None:
        assert path.suffix == ".mkv", "Video must be saved as MKV"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        images = [sample.image for sample in self.samples_list]
        save_lossless_mkv(images, path, fps=fps)

    def save_metadata(self, path: Path) -> None:
        assert path.suffix == ".json", "Metadata must be saved as JSON"
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(
                {
                    **asdict(self.metadata),
                    "timestamps": [s.metadata.timestamp for s in self.samples_list],
                    "detections": [
                        [det.to_dict() for det in s.metadata.detections] if s.metadata.detections else None
                        for s in self.samples_list
                    ],
                },
                f,
            )

    @staticmethod
    def video_path(path: Path) -> Path:
        return path / "video.mkv"

    @staticmethod
    def metadata_path(path: Path) -> Path:
        return path / "metadata.json"

    def save(self, path: Path, fps: int = 1) -> tuple[Path, Path]:
        path.mkdir(parents=True, exist_ok=True)
        video_path = self.video_path(path)
        metadata_path = self.metadata_path(path)
        self.save_video(video_path, fps=fps)
        self.save_metadata(metadata_path)
        return video_path, metadata_path

    @classmethod
    def load(cls, path: Path) -> "VideoSample":
        video_path = cls.video_path(path)
        metadata_path = cls.metadata_path(path)
        assert video_path.exists(), f"Video file does not exist at {video_path}"
        assert metadata_path.exists(), f"Metadata file does not exist at {metadata_path}"
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        timestamps = metadata.pop("timestamps")
        detections = metadata.pop("detections")
        if detections is not None:
            detections = [
                [Detection.from_dict(d) for d in frame_detections] if frame_detections else None
                for frame_detections in detections
            ]
        metadata = VideoMetadata(**metadata)
        images = read_mkv(video_path)
        assert len(images) == len(timestamps), "Number of frames and timestamps do not match"
        assert len(images) == len(detections), "Number of frames and detections do not match"
        samples = [
            ImageSample(
                image=img,
                metadata=ImageMetadata(frame_idx=i, timestamp=t, uuid=metadata.uuid, detections=d),
            )
            for i, (img, t, d) in enumerate(zip(images, timestamps, detections))
        ]
        return cls(samples=samples, metadata=metadata)

    def __repr__(self) -> str:
        num_detections = [len(s.metadata.detections or []) for s in self.samples_list]
        return f"VideoSample(samples={len(self.samples)}, detections={num_detections}, metadata={self.metadata})"
