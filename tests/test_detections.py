import unittest

import cv2
from ultralytics import YOLO


class TestDetections(unittest.TestCase):
    def test_detections(self):
        from easysort.utils.detections import Detection, Detections

        model = YOLO("__old__/_old/best.pt")
        image = cv2.imread("__old__/_old/test.jpg")
        detections = Detections.from_ultralytics(list(model(image))[0])
        assert isinstance(detections, list)
        assert isinstance(detections[0], Detection)

    def test_detection(self):
        from easysort.utils.detections import Detection

        detection1 = Detection(box=(100, 100, 200, 200), class_id=0, confidence=0.9)
        detection2 = Detection(box=(100, 100, 200, 200), class_id=0, confidence=0.9)
        assert detection1.box == detection2.box
        assert detection1.confidence == detection2.confidence
        assert len(detection1.center_point) == 3
        assert detection1.center_point == (150, 150, 0)
