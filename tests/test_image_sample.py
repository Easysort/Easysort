import json

import numpy as np
import pytest
from PIL import Image

from easysort.utils.detections import Detection
from easysort.utils.image_sample import ImageMetadata, ImageSample, VideoMetadata, VideoSample


@pytest.fixture
def sample_image():
    return Image.new("RGB", (10, 10), color="red")


@pytest.fixture
def sample_detection():
    return Detection(box=np.array([1, 1, 5, 5]), class_id=0, confidence=0.9, names={0: "test"})


@pytest.fixture
def sample_image_metadata():
    return ImageMetadata(frame_idx=0, timestamp=123.456, uuid="test-uuid", detections=None)


@pytest.fixture
def image_sample(sample_image, sample_detection, sample_image_metadata):
    metadata = sample_image_metadata
    metadata.detections = [sample_detection]
    return ImageSample(image=sample_image, metadata=metadata)


@pytest.fixture
def video_metadata():
    return VideoMetadata(date="2021-01-01", robot_id="1")


def test_image_sample_save_load(image_sample, tmp_path):
    # Test saving
    image_path = tmp_path / "test_image.png"
    metadata_path = tmp_path / "test_metadata.json"

    image_sample.save_image(image_path)
    image_sample.save_metadata(metadata_path)

    # Verify files exist
    assert image_path.exists()
    assert metadata_path.exists()

    # Test loading
    loaded_sample = ImageSample.load(image_path, metadata_path)

    # Check metadata was preserved
    assert loaded_sample.metadata.frame_idx == image_sample.metadata.frame_idx
    assert loaded_sample.metadata.timestamp == image_sample.metadata.timestamp
    assert loaded_sample.metadata.uuid == image_sample.metadata.uuid

    # Check metadata file content
    with open(metadata_path, "r") as f:
        metadata_content = json.load(f)
    assert metadata_content["frame_idx"] == 0
    assert metadata_content["timestamp"] == 123.456
    assert metadata_content["uuid"] == "test-uuid"


def test_video_sample_save_load(sample_image, sample_detection, video_metadata, tmp_path):
    # Create image samples with same UUID as video metadata
    image_metadata1 = ImageMetadata(
        frame_idx=0, timestamp=123.456, uuid=video_metadata.uuid, detections=[sample_detection]
    )
    image_metadata2 = ImageMetadata(
        frame_idx=1, timestamp=123.556, uuid=video_metadata.uuid, detections=[sample_detection]
    )

    sample1 = ImageSample(image=sample_image, metadata=image_metadata1)
    sample2 = ImageSample(image=sample_image, metadata=image_metadata2)

    # Create a video sample
    video_sample = VideoSample(samples=[sample1, sample2], metadata=video_metadata)

    # Save video and metadata
    video_path = tmp_path / "test_video.mkv"
    metadata_path = tmp_path / "test_video_metadata.json"

    video_sample.save_video(video_path, fps=10)
    video_sample.save_metadata(metadata_path)

    # Verify files exist
    assert video_path.exists()
    assert metadata_path.exists()

    # Test loading
    loaded_video_sample = VideoSample.load(video_path, metadata_path)

    # Check metadata was preserved
    assert loaded_video_sample.metadata.date == video_sample.metadata.date
    assert loaded_video_sample.metadata.robot_id == video_sample.metadata.robot_id
    assert loaded_video_sample.metadata.uuid == video_sample.metadata.uuid

    # Check samples were preserved
    assert len(loaded_video_sample.samples) == len(video_sample.samples)
    assert 0 in loaded_video_sample.samples
    assert 1 in loaded_video_sample.samples

    # Check timestamps and frame indices
    assert loaded_video_sample.samples[0].metadata.timestamp == sample1.metadata.timestamp
    assert loaded_video_sample.samples[1].metadata.timestamp == sample2.metadata.timestamp

    # Check detection content
    assert sample1.metadata.detections is not None
    assert sample2.metadata.detections is not None
    assert loaded_video_sample.samples[0].metadata.detections is not None
    assert loaded_video_sample.samples[1].metadata.detections is not None
    assert all(a == b for a, b in zip(loaded_video_sample.samples[0].metadata.detections, sample1.metadata.detections))
    assert all(a == b for a, b in zip(loaded_video_sample.samples[1].metadata.detections, sample2.metadata.detections))


def test_video_sample_samples_list(sample_image, sample_detection, video_metadata):
    # Create samples with non-sequential frame indices
    image_metadata1 = ImageMetadata(
        frame_idx=3, timestamp=123.456, uuid=video_metadata.uuid, detections=[sample_detection]
    )
    image_metadata2 = ImageMetadata(
        frame_idx=1, timestamp=123.556, uuid=video_metadata.uuid, detections=[sample_detection]
    )

    sample1 = ImageSample(image=sample_image, metadata=image_metadata1)
    sample2 = ImageSample(image=sample_image, metadata=image_metadata2)

    video_sample = VideoSample(samples=[sample1, sample2], metadata=video_metadata)

    # Test samples_list returns samples in order of frame_idx
    samples_list = video_sample.samples_list
    assert len(samples_list) == 2
    assert samples_list[0].metadata.frame_idx == 1  # First should be frame 1
    assert samples_list[1].metadata.frame_idx == 3  # Second should be frame 3
