from easysort.sorting.infer_yolov8_ultralytics import Classifier
from easysort.sorting.infer_yoloWorld import ClassifierYoloWorld, test_yoloworld_classes, class_to_fraction_mapping
from easysort.sorting.segmentation_fastsam import Segmentation
from easysort.utils.detections import Detection
from easysort.common.environment import Environment
from easysort.system.camera.realsense_connector import RealSenseConnector
from easysort.system.camera.camera_connector import CameraConnector
from easysort.visualize.helpers import visualize_sorting_pipeline_image
from easysort.common.image_registry import ImageRegistry
from easysort.utils.image_sample import VideoMetadata
from easysort.system.gantry.connector import GantryConnector
from easysort.utils.gantry_math import realsense2gantry, detection_to_pickup, gantryExtrapolate1point5sec
from easysort.common.logger import EasySortLogger

import numpy as np
from typing import List, Optional
import cv2
import time
from datetime import datetime

_LOGGER = EasySortLogger(__name__)

CENTER_POINT = (-55, 0, 0)

class SortingPipeline:
    def __init__(self, use_yolo_world: bool = False):
        self.camera = RealSenseConnector()# if not Environment.DEBUG else CameraConnector()
        self.classifier = ClassifierYoloWorld(test_yoloworld_classes) if use_yolo_world else Classifier()
        # self.segmentation = Segmentation()
        self.image_registry = ImageRegistry()
        self.connector = GantryConnector(Environment.GANTRY_PORT, self.camera)
        self.connector.go_to(CENTER_POINT[0], CENTER_POINT[1], CENTER_POINT[2])
        while not self.connector.is_ready: pass
        _LOGGER.info("Connector ready")
    
    def __call__(self, image: np.ndarray, timestamp: Optional[float] = None) -> List[Detection]:
        if timestamp is None: timestamp = time.time()
        detections = self.classifier(image)
        # detections = self.segmentation(image, detections)
        for detection in detections:
            detection.timestamp = timestamp
        return detections

    def stream(self):
        metadata = VideoMetadata(date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), robot_id=Environment.CURRENT_ROBOT_ID)
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
    
    def stream_pickup(self):
        metadata = VideoMetadata(date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), robot_id=Environment.CURRENT_ROBOT_ID)
        self.image_registry.set_video_metadata(metadata)
        while True:
            color_image, timestamp = self.camera.get_color_image()
            self.image_registry.add(color_image, timestamp=timestamp)
            start_time = time.time()
            print("I got the image")
            detections = self(color_image, timestamp)
            print("I got the detections and I took")
            print(time.time() - start_time)
            if time.time() - start_time < 0.35:
                amount = 0.35 - (time.time() - start_time)
                print(f"Sleeping for {amount} seconds")
                time.sleep(amount)
            for d in detections:
                d.set_center_point((d.center_point[0], d.center_point[1], self.camera.get_depth_for_detection(d)))
            if len(detections) > 0:
                print("---- PICKUP ----")
                print(detections)
                pickup_detection = detection_to_pickup(detections)
                if pickup_detection is None:
                    print("No pickup detection found")
                    continue
                print(pickup_detection.center_point)
                if pickup_detection.center_point[2] > 0:
                    gantry_point = realsense2gantry(np.array([pickup_detection.center_point[0], pickup_detection.center_point[1]]), pickup_detection.center_point[2])
                    print(gantry_point)
                    gantry_point_after_2sec = gantryExtrapolate1point5sec(gantry_point)
                    gantry_point_tuple = (gantry_point_after_2sec[0], gantry_point_after_2sec[1], gantry_point_after_2sec[2])
                    self.connector.pickup(gantry_point_tuple, class_to_fraction_mapping[pickup_detection.class_name], CENTER_POINT)
                print("---- END PICKUP ----")
                while not self.connector.is_ready: pass
                yield pickup_detection
            if Environment.DEBUG:
                main_view = visualize_sorting_pipeline_image(color_image, detections, show_plot=False)
                cv2.imshow("Main View", main_view)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.connector.quit()
                    break

if __name__ == "__main__":
    # Has to be run with sudo if DEBUG is False
    pipeline = SortingPipeline(use_yolo_world=True)
    for detection in pipeline.stream_pickup():
        print("------PICKED UP DETECTION--------")
        print(detection)
        print("--------------------------------")
