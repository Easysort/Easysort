
from easysort.sorting.infer_yolov8_ultralytics import Classifier
from easysort.sorting.segmentation_fastsam import Segmentation
from easysort.utils.detections import Detection
from easysort.common.environment import Environment
from easysort.system.camera.realsense_connector import RealSenseConnector
from easysort.visualize.helpers import visualize_sorting_pipeline_image
from easysort.common.image_registry import ImageRegistry
from easysort.utils.image_sample import VideoMetadata

import numpy as np
from typing import List, Optional
import cv2
import time
from datetime import datetime

class SortingPipeline:
    def __init__(self):
        self.camera = RealSenseConnector()
        self.classifier = Classifier()
        self.segmentation = Segmentation()
        self.image_registry = ImageRegistry()

    def __call__(self, image: np.ndarray, timestamp: Optional[float] = None) -> List[Detection]:
        if timestamp is None: timestamp = time.time()
        detections = self.classifier(image)
        detections = self.segmentation(image, detections)
        for detection in detections:
            detection.timestamp = timestamp
        return detections

    def stream(self):
        metadata = VideoMetadata(date=datetime.now().strftime("%Y-%m-%d"), robot_id=Environment.CURRENT_ROBOT_ID)
        self.image_registry.set_video_metadata(metadata)
        while True:
            color_image, timestamp = self.camera.get_color_image()
            self.image_registry.add(color_image, timestamp=timestamp)
            detections = self(color_image, timestamp)
            for d in detections:
                d.set_center_point((d.center_point[0], d.center_point[1], self.camera.get_depth_for_detection(d)))
            yield detections
            if Environment.DEBUG:
                main_view = visualize_sorting_pipeline_image(color_image, detections, show_plot=False)
                cv2.imshow("Main View", main_view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    # Has to be run with sudo
    pipeline = SortingPipeline()
    for detections in pipeline.stream():
        print(detections)
        print("--------------------------------")

    # To run image:
    # SOURCE_IMAGE_PATH = "__old__/_old/test.jpg"
    # image = cv2.imread(SOURCE_IMAGE_PATH)
    # detections = pipeline(image)
    # pipeline.visualize(image, detections)
