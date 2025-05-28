import os
import time
import unittest

import numpy as np
from PIL import Image


@unittest.skipIf(os.getenv("DEEP_TEST") is None, "Skipping deep test")
class TestImageRegistry(unittest.TestCase):
    def test_image_registry_connection(self):
        from easysort.common.environment import Environment
        from easysort.common.image_registry import SupabaseHelper
        from easysort.utils.detections import Detection
        from easysort.utils.image_sample import ImageMetadata, ImageSample, VideoMetadata, VideoSample

        helper = SupabaseHelper(Environment.SUPABASE_AI_IMAGES_BUCKET)
        metadata = VideoMetadata(date="2021-01-01", robot_id="1")
        image_sample = ImageSample(
            image=Image.open("__old__/_old/test.jpg"),
            detections=[Detection(box=np.array([10, 10, 20, 20]), class_id=0, confidence=0.5, names={0: "test"})],
            metadata=ImageMetadata(frame_idx=0, timestamp=0, uuid=metadata.uuid),
        )
        image_sample2 = ImageSample(
            image=Image.open("__old__/_old/test.jpg"),
            detections=[
                Detection(box=np.array([10, 10, 20, 20]), class_id=0, confidence=0.5, names={0: "test"}),
                Detection(box=np.array([10, 10, 20, 20]), class_id=0, confidence=0.5, names={0: "test"}),
            ],
            metadata=ImageMetadata(frame_idx=1, timestamp=1, uuid=metadata.uuid),
        )

        sample = VideoSample(samples=[image_sample, image_sample2], metadata=metadata)
        helper.upload_sample(sample)
        sample2 = helper.get(metadata.uuid)
        helper.delete(metadata.uuid)
        self.assertEqual(sample.metadata.uuid, sample2.metadata.uuid)
        self.assertEqual(len(sample.samples), len(sample2.samples))
        self.assertEqual(len(sample.samples[0].detections), len(sample2.samples[0].detections))
        self.assertEqual(len(sample.samples[1].detections), len(sample2.samples[1].detections))

    def test_image_registry_class(self):
        from easysort.common.environment import Environment
        from easysort.common.image_registry import ImageRegistry
        from easysort.utils.image_sample import VideoMetadata, VideoSample

        image_registry = ImageRegistry()

        metadata = VideoMetadata(date="2021-01-01", robot_id="1")
        image_registry.set_video_metadata(metadata)
        timestamp = time.time()
        image = Image.open("__old__/_old/test.jpg")
        image_registry.add(image, timestamp=timestamp)
        image_registry.add(image, timestamp=timestamp)
        image_registry.add(image, timestamp=timestamp)
        image_registry.add(image, timestamp=timestamp)
        image_registry.add(image, timestamp=timestamp)

        assert len(os.listdir(os.path.join(Environment.IMAGE_REGISTRY_PATH, metadata.uuid))) == 6
        assert os.path.exists(os.path.join(Environment.IMAGE_REGISTRY_PATH, metadata.uuid, "metadata.json"))
        video_sample = image_registry.compress_image_samples_to_video(metadata.uuid)

        assert isinstance(video_sample, VideoSample)
        assert len(video_sample.samples) == 5
        assert video_sample.metadata.uuid == metadata.uuid
        assert video_sample.metadata.date == metadata.date
        assert video_sample.metadata.robot_id == metadata.robot_id
        assert not os.path.exists(os.path.join(Environment.IMAGE_REGISTRY_PATH, metadata.uuid))


if __name__ == "__main__":
    unittest.main()
