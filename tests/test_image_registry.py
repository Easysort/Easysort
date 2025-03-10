
import unittest
import os
from PIL import Image
import numpy as np

from easysort.common.image_registry import SupabaseHelper
from easysort.utils.image_sample import DetectionSample, DetectionMetadata
from easysort.utils.detections import Detection


@unittest.skipIf(os.getenv("DEEP_TEST") is None, "Skipping deep test")
class TestImageRegistry(unittest.TestCase):
    def test_image_registry_connection(self):
        helper = SupabaseHelper("ai-images")
        sample = DetectionSample(
            images=[Image.open("__old__/_old/test.jpg"), Image.open("__old__/_old/test.jpg")],
            detections=[[Detection(box=np.array([10, 10, 20, 20]), class_id=0, conf=0.5, names=["test"])], []],
            metadata=DetectionMetadata(uuid="1", date="2021-01-01", robot_id="1")
        )
        helper.upload_sample(sample)
        sample2 = helper.get("1")
        helper.delete("1")
        self.assertEqual(sample.metadata.uuid, sample2.metadata.uuid)
        self.assertEqual(len(sample.images), len(sample2.images))
        self.assertEqual(len(sample.detections), len(sample2.detections))

if __name__ == "__main__":
    unittest.main()