
from easysort.system.camera.realsense_connector import RealSenseConnector
from easysort.common.image_registry import ImageRegistry
from easysort.common.environment import Environment
from easysort.utils.image_sample import VideoMetadata
from datetime import datetime

import time
import cv2

class Observer:
    def __init__(self):
        self.camera = RealSenseConnector()
        self.image_registry = ImageRegistry()

    def observe_single_class(self, type: str):
        metadata = VideoMetadata(date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), robot_id=Environment.CURRENT_ROBOT_ID)
        metadata.uuid = metadata.uuid + "_" + type
        self.image_registry.set_video_metadata(metadata)
        while True:
            color_image, timestamp = self.camera.get_color_image()
            self.image_registry.add(color_image, timestamp=timestamp)
            cv2.imshow("Main View", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.25)

if __name__ == "__main__":
    observer = Observer()
    CURRENT_TYPE = "cardboard"
    observer.observe_single_class(CURRENT_TYPE)