import time
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from easysort.common.environment import Environment
from easysort.common.image_registry import ImageRegistry
from easysort.sorting.infer_yolov8_ultralytics import Classifier
from easysort.sorting.infer_yoloWorld import ClassifierYoloWorld, test_yoloworld_classes
from easysort.system.camera.camera_connector import CameraConnector
from easysort.utils.detections import Detection
from easysort.utils.image_sample import VideoMetadata
from easysort.visualize.helpers import visualize_sorting_pipeline_image


class SortingPipeline:
    def __init__(self, use_yolo_world: bool = False):
        self.camera = CameraConnector()
        self.classifier = ClassifierYoloWorld(test_yoloworld_classes) if use_yolo_world else Classifier()
        # self.segmentation = Segmentation()
        self.image_registry = ImageRegistry()

    def __call__(self, image: np.ndarray, timestamp: Optional[float] = None) -> List[Detection]:
        if timestamp is None:
            timestamp = time.time()
        detections = self.classifier(image)
        # detections = self.segmentation(image, detections)
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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def stream_pickup(self):
        metadata = VideoMetadata(date=datetime.now().strftime("%Y-%m-%d"), robot_id=Environment.CURRENT_ROBOT_ID)
        self.image_registry.set_video_metadata(metadata)
        while True:
            color_image, timestamp = self.camera.get_color_image()
            self.image_registry.add(color_image, timestamp=timestamp)
            detections = self(color_image, timestamp)
            for d in detections:
                d.set_center_point((d.center_point[0], d.center_point[1], self.camera.get_depth_for_detection(d)))
            yield detections[0]
            # if len(detections) > 0:
            #     self.connector.pickup_detection(self.transform_camera_to_robot(detections[0]))
            #     while not self.connector.is_ready: pass
            #     yield detections[0]


if __name__ == "__main__":
    # Has to be run with sudo if DEBUG is False
    pipeline = SortingPipeline(use_yolo_world=True)
    for detection in pipeline.stream():
        print("------PICKED UP DETECTION--------")
        print(detection)
        print("--------------------------------")
