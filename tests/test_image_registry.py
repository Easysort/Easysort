
import unittest
import os
from PIL import Image
import numpy as np

@unittest.skipIf(os.getenv("DEEP_TEST") is None, "Skipping deep test")
class TestImageRegistry(unittest.TestCase):
    def test_image_registry_connection(self):
        from easysort.common.image_registry import SupabaseHelper
        from easysort.utils.image_sample import VideoSample, VideoMetadata, ImageSample, ImageMetadata
        from easysort.utils.detections import Detection
        from easysort.common.environment import Environment
        helper = SupabaseHelper(Environment.SUPABASE_AI_IMAGES_BUCKET)
        metadata = VideoMetadata(date="2021-01-01", robot_id="1")
        image_sample = ImageSample(
            image=Image.open("__old__/_old/test.jpg"),
            detections=[Detection(box=np.array([10, 10, 20, 20]), class_id=0, conf=0.5, names=["test"])],
            metadata=ImageMetadata(frame_idx=0, timestamp=0, uuid=metadata.uuid)
        )
        image_sample2 = ImageSample(
            image=Image.open("__old__/_old/test.jpg"),
            detections=[
                Detection(box=np.array([10, 10, 20, 20]), class_id=0, conf=0.5, names=["test"]),
                Detection(box=np.array([10, 10, 20, 20]), class_id=0, conf=0.5, names=["test"])
            ],
            metadata=ImageMetadata(frame_idx=1, timestamp=1, uuid=metadata.uuid)
        )

        sample = VideoSample(
            samples=[image_sample, image_sample2],
            metadata=metadata
        )
        helper.upload_sample(sample)
        sample2 = helper.get(metadata.uuid)
        helper.delete(metadata.uuid)
        self.assertEqual(sample.metadata.uuid, sample2.metadata.uuid)
        self.assertEqual(len(sample.samples), len(sample2.samples))
        self.assertEqual(len(sample.samples[0].detections), len(sample2.samples[0].detections))
        self.assertEqual(len(sample.samples[1].detections), len(sample2.samples[1].detections))

    def test_image_registry_class(self):
        from easysort.common.image_registry import ImageRegistry
        _ = ImageRegistry()

if __name__ == "__main__":
    unittest.main()