import os
import time

import numpy as np
import pytest
from PIL import Image

from easysort.common.image_registry import Environment, ImageRegistry, SupabaseHelper
from easysort.utils.detections import Detection
from easysort.utils.image_sample import ImageMetadata, ImageSample, VideoMetadata, VideoSample


@pytest.fixture
def image_sample():
    return ImageSample(
        image=Image.new("RGB", (10, 10), color="red"),
        metadata=ImageMetadata(
            frame_idx=0,
            timestamp=123.456,
            uuid="test-uuid",
            detections=[Detection(box=np.array([1, 1, 5, 5]), class_id=0, confidence=0.9, names={0: "test"})],
        ),
    )


@pytest.mark.skipif(os.getenv("DEEP_TEST") is None, reason="Skipping deep test")
def test_supabase_connection(image_sample):
    helper = SupabaseHelper(Environment.SUPABASE_AI_IMAGES_BUCKET)
    metadata = VideoMetadata(date="2021-01-01", robot_id="1")

    sample = VideoSample(samples=[image_sample, image_sample], metadata=metadata)
    helper.upload_sample(sample)
    sample2 = helper.get(metadata.uuid)
    helper.delete(metadata.uuid)
    assert sample.metadata.uuid == sample2.metadata.uuid
    assert len(sample.samples) == len(sample2.samples)
    assert sample.samples[0].metadata.detections is not None
    assert sample.samples[1].metadata.detections is not None
    assert sample2.samples[0].metadata.detections is not None
    assert sample2.samples[1].metadata.detections is not None
    assert len(sample.samples[0].metadata.detections) == len(sample2.samples[0].metadata.detections)
    assert len(sample.samples[1].metadata.detections) == len(sample2.samples[1].metadata.detections)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Environment, "IMAGE_REGISTRY_PATH", tmp_path)
    return tmp_path


@pytest.fixture()
def tmp_registry(tmp_env):
    reg = ImageRegistry()
    metadata = VideoMetadata(date="2021-01-01", robot_id="1")
    reg.set_video_metadata(metadata)
    return reg, metadata


def test_image_registry(tmp_registry, image_sample):
    image_registry, metadata = tmp_registry

    timestamp = time.time()
    for i in range(10):
        image_registry.add(image_sample.image, timestamp=timestamp, detections=image_sample.metadata.detections)

    assert set(image_registry.save_path.glob("*.json")) == {
        image_registry.save_path / f"{i}.json" for i in range(10)
    } | {image_registry.save_path / "metadata.json"}
    assert set(image_registry.save_path.glob("*.png")) == {image_registry.save_path / f"{i}.png" for i in range(10)}
    assert (image_registry.save_path / "metadata.json").exists()
    video_sample = image_registry.convert_to_video(metadata.uuid)

    assert isinstance(video_sample, VideoSample)
    assert len(video_sample.samples) == 10
    assert video_sample.metadata.uuid == metadata.uuid
    assert video_sample.metadata.date == metadata.date
    assert video_sample.metadata.robot_id == metadata.robot_id
