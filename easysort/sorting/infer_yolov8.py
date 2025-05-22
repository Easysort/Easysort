# pylint: disable=no-name-in-module
from tinygrad.examples.yolov8 import YOLOv8  # pylint: disable=no-name-in-module
from tinygrad.tinygrad import Tensor  # pylint: disable=no-name-in-module

import cv2
import supervision as sv  # type: ignore

from easysort.common.logger import EasySortLogger
import time

LOGGER = EasySortLogger()
RANDOM_IMAGE_TENSOR = Tensor.rand((980, 1280, 3))


def load_model_params():
    return 0, 0, 0, 0


class Classifier:
    def __init__(self):
        w, r, d, num_classes = load_model_params()
        self.model = YOLOv8(w, r, d, num_classes)
        LOGGER.info("Classifier initialized")

    def __call__(self, image):
        results = self.model(image)
        results_unlisted = list(results)[
            0
        ]  # You can pass multiple images, we have one, so we take the first results object.
        return results_unlisted

    def test_speed(self) -> None:
        time0 = time.time()
        self(RANDOM_IMAGE_TENSOR)
        LOGGER.info(f"Time taken: {round(time.time() - time0, 2)} seconds")

    def visualize(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        detections = self(image)
        sv.plot_image(sv.BoundingBoxAnnotator(thickness=2).annotate(image, detections), (10, 10))

    def cam_view_to_world_view(self, detections):
        # Do computations...
        return detections


if __name__ == "__main__":
    SOURCE_IMAGE_PATH = "_old/test.jpg"
    image = cv2.imread(SOURCE_IMAGE_PATH)
    print(image.shape)
    classifier = Classifier()
    detections = classifier(image)
    annotated_image = image.copy()

    # PLOT ANNOTATIONS
