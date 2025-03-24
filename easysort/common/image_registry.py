import supabase
import os 
from PIL import Image
from typing import List, Optional
import json
from pathlib import Path
import numpy as np
from dataclasses import asdict
import storage3.exceptions

from easysort.utils.detections import Detection
from easysort.utils.image_sample import VideoSample, ImageSample, VideoMetadata, ImageMetadata
from easysort.common.environment import Environment


class ImageRegistry:
    def __init__(self) -> None:
        self.supabase_helper = SupabaseHelper(Environment.SUPABASE_AI_IMAGES_BUCKET)

    def set_video_metadata(self, metadata: VideoMetadata) -> None:
        self.video_metadata = metadata
        self.frame_idx = 0

    def save_video_metadata(self) -> None:
        path = Path(os.path.join(Environment.IMAGE_REGISTRY_PATH, self.video_metadata.uuid, "metadata.json"))
        if path.exists(): return
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f: json.dump(asdict(self.video_metadata), f)

    def add(self, image: Image.Image | np.ndarray, timestamp: float, detections: Optional[List[Detection]] = None) -> None: # Saves locally
        if isinstance(image, np.ndarray): image = Image.fromarray(image)
        if detections is None: detections = []
        sample = ImageSample(image, detections, ImageMetadata(self.frame_idx, timestamp, self.video_metadata.uuid))
        path = Path(os.path.join(Environment.IMAGE_REGISTRY_PATH, self.video_metadata.uuid, f"{self.frame_idx}.sample"))
        os.makedirs(path.parent, exist_ok=True)
        with path.open("w") as f: json.dump(sample.to_json(), f)
        self.save_video_metadata()
        self.frame_idx += 1

    def compress_image_samples_to_video(self, uuid: str, delete: Optional[bool] = True) -> VideoSample:
        image_paths = list(Path(os.path.join(Environment.IMAGE_REGISTRY_PATH, uuid)).glob("*.sample"))
        image_paths.sort(key=lambda x: int(x.stem))
        samples = [ImageSample.from_json(json.load(open(path))) for path in image_paths]
        metadata = VideoMetadata(**json.load(open(os.path.join(Environment.IMAGE_REGISTRY_PATH, uuid, "metadata.json"))))
        video_sample = VideoSample(samples, metadata)
        if delete:
            for path in Path(os.path.join(Environment.IMAGE_REGISTRY_PATH, uuid)).glob("*"): path.unlink()
            os.rmdir(os.path.join(Environment.IMAGE_REGISTRY_PATH, uuid))
        return video_sample

    def upload(self) -> None: # Uploads images to Supabase and deletes them locally
        uuids = set([p.parts[1] for p in Path(Environment.IMAGE_REGISTRY_PATH).glob("**/*.sample")])
        for uuid in uuids:
            video_sample = self.compress_image_samples_to_video(uuid)
            self.supabase_helper.upload_sample(video_sample)

class SupabaseHelper:
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name: str = bucket_name
        self.client = supabase.create_client(Environment.SUPABASE_URL, Environment.SUPABASE_KEY)

    def upload_sample(self, sample: VideoSample) -> None:
        file_obj = sample.to_json().encode('utf-8')
        self.client.storage.from_(self.bucket_name).upload(
            path=f"{sample.metadata.uuid}.json",
            file=file_obj,
            file_options={
                "content-type": "application/json",
                "cache-control": "3600",
                "upsert": "false"  # False = Error if file already exists
            }
        )

    def exists(self, uuid: str) -> bool:
        try:
            self.client.storage.from_(self.bucket_name).download(f"{uuid}.json")
            return True
        except storage3.exceptions.StorageApiError: return False

    def get(self, uuid: str) -> VideoSample:
        response = self.client.storage.from_(self.bucket_name).download(f"{uuid}.json")
        json_str = response.decode('utf-8')
        return VideoSample.from_json(json_str)

    def delete(self, uuid: str) -> None:
        self.client.storage.from_(self.bucket_name).remove(f"{uuid}.json")

if __name__ == "__main__":
    image_registry = ImageRegistry()
    image_registry.upload()