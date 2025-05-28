import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import numpy as np
import storage3.exceptions
import supabase
from PIL import Image

from easysort.common.environment import Environment
from easysort.utils.detections import Detection
from easysort.utils.image_sample import ImageMetadata, ImageSample, VideoMetadata, VideoSample


class ImageRegistry:
    def __init__(self) -> None:
        self.video_metadata: Optional[VideoMetadata] = None
        self._supabase_helper: "SupabaseHelper | None" = None

    def set_video_metadata(self, metadata: VideoMetadata) -> None:
        self.video_metadata = metadata
        self.frame_idx = 0
        self.save_path.mkdir(parents=True, exist_ok=True)

    @property
    def uuid(self) -> str:
        assert self.video_metadata is not None, "Video metadata not set, please call set_video_metadata first"
        return self.video_metadata.uuid

    @property
    def save_path(self) -> Path:
        return Environment.IMAGE_REGISTRY_PATH / self.uuid

    @property
    def supabase_helper(self) -> "SupabaseHelper":
        if self._supabase_helper is None:
            self._supabase_helper = SupabaseHelper(Environment.SUPABASE_AI_IMAGES_BUCKET)
        return self._supabase_helper

    def save_video_metadata(self) -> None:
        assert self.video_metadata is not None, "Video metadata not set, please call set_video_metadata first"
        path = self.save_path / "metadata.json"
        if path.exists():
            return
        with open(path, "w") as f:
            json.dump(asdict(self.video_metadata), f)

    def add(
        self, image: Image.Image | np.ndarray, timestamp: float, detections: Optional[List[Detection]] = None
    ) -> None:  # Saves locally
        assert self.video_metadata is not None, "Video metadata not set, please call set_video_metadata first"
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if detections is None:
            detections = []
        sample = ImageSample(image, ImageMetadata(self.frame_idx, timestamp, self.uuid, detections))
        metadata_path = self.save_path / f"{self.frame_idx}.json"
        image_path = self.save_path / f"{self.frame_idx}.png"
        sample.save_metadata(metadata_path)
        sample.save_image(image_path)
        self.save_video_metadata()
        self.frame_idx += 1

    @staticmethod
    def exists(uuid: str) -> bool:
        return (Environment.IMAGE_REGISTRY_PATH / uuid).exists()

    @staticmethod
    def uuids() -> set[str]:
        """
        Returns a list of all UUIDs in the image registry.
        """
        return set([p.parent.name for p in Environment.IMAGE_REGISTRY_PATH.glob("**/*.json")])

    @staticmethod
    def cleanup(min_len: int = 10) -> None:
        """
        Cleans up the image registry by removing directories that contain fewer than `min_len` samples.
        """
        for uuid in ImageRegistry.uuids():
            path = Environment.IMAGE_REGISTRY_PATH / uuid
            if len(list(path.glob("*.json"))) < min_len:
                shutil.rmtree(path)

    @staticmethod
    def convert_to_video(uuid: str) -> VideoSample:
        path = Environment.IMAGE_REGISTRY_PATH / uuid
        metadata = VideoMetadata(**json.load(open(path / "metadata.json")))
        image_paths = list(sorted(path.glob("*.png"), key=lambda x: int(x.stem)))
        json_paths = [path / f"{p.stem}.json" for p in image_paths]
        samples = []
        for image_path, json_path in zip(image_paths, json_paths):
            assert image_path.exists(), f"Image {image_path} does not exist"
            assert json_path.exists(), f"Metadata {json_path} does not exist"
            sample = ImageSample.load(image_path=image_path, metadata_path=json_path)
            samples.append(sample)
        return VideoSample(samples, metadata)

    def upload(self, uuid: str) -> None:
        """
        Uploads a single video sample to Supabase.
        """
        video_sample = self.convert_to_video(uuid)
        self.supabase_helper.upload_sample(video_sample)

    def upload_all(self, delete: bool = False) -> None:
        for uuid in ImageRegistry.uuids():
            self.upload(uuid)
            if delete:
                shutil.rmtree(Environment.IMAGE_REGISTRY_PATH / uuid)


class SupabaseHelper:
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name: str = bucket_name
        self.client = supabase.create_client(Environment.SUPABASE_URL, Environment.SUPABASE_KEY)
        self.bucket = self.client.storage.from_(self.bucket_name)

    def upload_sample(self, sample: VideoSample) -> None:
        with TemporaryDirectory() as tmpdir:
            video_path, metadata_path = sample.save(Path(tmpdir), fps=1)
            self.bucket.upload(
                path=f"{self.bucket_name}/{sample.metadata.uuid}.json",
                file=metadata_path,
                file_options={
                    "content-type": "application/json",
                    "cache-control": "3600",
                    "upsert": "false",  # False = Error if file already exists
                },
            )
            self.bucket.upload(
                path=f"{self.bucket_name}/{sample.metadata.uuid}.mkv",
                file=video_path,
                file_options={
                    "content-type": "video/matroska",
                    "cache-control": "3600",
                    "upsert": "false",  # False = Error if file already exists
                },
            )

    def exists(self, uuid: str) -> bool:
        try:
            self.bucket.download(f"{self.bucket_name}/{uuid}.json")
            return True
        except storage3.exceptions.StorageApiError:
            return False

    def get(self, uuid: str) -> VideoSample:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            video_path = VideoSample.video_path(path)
            metadata_path = VideoSample.metadata_path(path)
            with open(metadata_path, "wb") as f:
                f.write(self.bucket.download(f"{self.bucket_name}/{uuid}.json"))
            with open(video_path, "wb") as f:
                f.write(self.bucket.download(f"{self.bucket_name}/{uuid}.mkv"))
            return VideoSample.load(path)

    def delete(self, uuid: str) -> None:
        self.bucket.remove([f"{self.bucket_name}/{uuid}.json"])
        self.bucket.remove([f"{self.bucket_name}/{uuid}.mkv"])


if __name__ == "__main__":
    image_registry = ImageRegistry()
    # image_registry.upload_all()
    image_registry.cleanup()
