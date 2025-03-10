import cv2
import supervision as sv

from easysort.common.logger import EasySortLogger
from easysort.common.timer import TimeIt
from easysort.utils.detections import Detection
from ultralytics import YOLO
import time
import torch
from typing import List
import numpy as np
LOGGER = EasySortLogger()
RANDOM_IMAGE_TENSOR = torch.rand((980, 1280, 3))

class Classifier:
    def __init__(self):
        self.model = YOLO("/Users/lucasvilsen/Documents/Documents/EasySort/__old__/_old/runs/train4/weights/best.pt")
        LOGGER.info("Classifier initialized")

    @TimeIt("Bbox classification")
    def __call__(self, image) -> List[Detection]:
        results = self.model(image, stream=True)
        results_unlisted = list(results)[0] # You can pass multiple images, we have one, so we take the first results object.
        return Detection.from_ultralytics(results_unlisted)

    def test_speed(self) -> None:
        time0 = time.time()
        self(RANDOM_IMAGE_TENSOR)
        LOGGER.info(f"Time taken: {round(time.time() - time0, 2)} seconds")

    def visualize(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        detections = self(image)
        sv.plot_image(sv.BoundingBoxAnnotator(thickness=2).annotate(image, detections), (10, 10))

    def cam_view_to_world_view(self, detections):
        # Do computations hehe..
        return detections

if __name__ == "__main__":
    SOURCE_IMAGE_PATH = "__old__/_old/test.jpg"
    image = cv2.imread(SOURCE_IMAGE_PATH)
    classifier = Classifier()
    detections = classifier(image)
    annotated_image = image.copy()

    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

    detections_formatted = sv.Detections(
        xyxy=np.array([d.box for d in detections]),
        class_id=np.array([d.class_id for d in detections]),
        confidence=np.array([d.confidence for d in detections])
    )

    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections_formatted)
    annotated_image = LABEL_ANNOTATOR.annotate(
        annotated_image, detections_formatted,
        labels=[detections[0].names[class_id] for class_id in detections_formatted.class_id]
    )
    sv.plot_image(annotated_image)

