import unittest

import numpy as np
from PIL import Image

from easysort.utils.detections import Detection
from easysort.utils.image_sample import ImageMetadata, ImageSample, VideoMetadata, VideoSample


class TestImageSample(unittest.TestCase):
    def test_image_sample_to_json(self):
        image = Image.new("RGB", (10, 10), color="red")
        detections = [Detection(box=np.array([1, 1, 5, 5]), class_id=0, confidence=0.9, names={0: "test"})]
        metadata = ImageMetadata(frame_idx=0, timestamp=123.456, uuid="test-uuid")
        sample = ImageSample(image=image, detections=detections, metadata=metadata)
        json_data = sample.to_json()
        self.assertIn('"image":', json_data)
        self.assertIn('"detections":', json_data)
        self.assertIn('"metadata":', json_data)

    def test_image_sample_from_json(self):
        image = Image.new("RGB", (10, 10), color="red")
        detections = [Detection(box=np.array([1, 1, 5, 5]), class_id=0, confidence=0.9, names={0: "test"})]
        metadata = ImageMetadata(frame_idx=0, timestamp=123.456, uuid="test-uuid")
        sample = ImageSample(image=image, detections=detections, metadata=metadata)
        json_data = sample.to_json()
        new_sample = ImageSample.from_json(json_data)
        self.assertEqual(new_sample.metadata.frame_idx, sample.metadata.frame_idx)
        self.assertEqual(new_sample.metadata.timestamp, sample.metadata.timestamp)
        self.assertEqual(new_sample.metadata.uuid, sample.metadata.uuid)
        self.assertEqual(len(new_sample.detections), len(sample.detections))

    def test_video_sample_to_json(self):
        image = Image.new("RGB", (10, 10), color="red")
        detections = [Detection(box=np.array([1, 1, 5, 5]), class_id=0, confidence=0.9, names={0: "test"})]
        metadata = VideoMetadata(date="2021-01-01", robot_id="1")
        image_metadata = ImageMetadata(frame_idx=0, timestamp=123.456, uuid=metadata.uuid)
        sample = ImageSample(image=image, detections=detections, metadata=image_metadata)
        video_sample = VideoSample(samples=[sample], metadata=metadata)
        json_data = video_sample.to_json()
        self.assertIn('"images":', json_data)
        self.assertIn('"metadata":', json_data)

    def test_video_sample_from_json(self):
        image = Image.new("RGB", (10, 10), color="red")
        detections = [Detection(box=np.array([1, 1, 5, 5]), class_id=0, confidence=0.9, names={0: "test"})]
        metadata = VideoMetadata(date="2021-01-01", robot_id="1")
        image_metadata = ImageMetadata(frame_idx=0, timestamp=123.456, uuid=metadata.uuid)
        sample = ImageSample(image=image, detections=detections, metadata=image_metadata)
        video_sample = VideoSample(samples=[sample], metadata=metadata)
        json_data = video_sample.to_json()
        new_video_sample = VideoSample.from_json(json_data)
        self.assertEqual(new_video_sample.metadata.date, video_sample.metadata.date)
        self.assertEqual(new_video_sample.metadata.robot_id, video_sample.metadata.robot_id)
        self.assertEqual(len(new_video_sample.samples), len(video_sample.samples))
        self.assertEqual(new_video_sample.samples[0].metadata.frame_idx, video_sample.samples[0].metadata.frame_idx)
        self.assertEqual(len(new_video_sample.samples[0].detections), len(video_sample.samples[0].detections))
