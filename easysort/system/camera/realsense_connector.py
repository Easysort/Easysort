from easysort.utils.detections import Detection
from typing import Tuple
import time
import numpy as np
import pyrealsense2 as rs # type: ignore
import os
import sys
from collections import OrderedDict

f_x = 635.47505771
f_y = 636.69651825
c_x = 323.90900467
c_y = 244.25395186

def get_pipeline():
    # On Mac the Intel Realsense library is extremely bad.
    # Connection only works 1 out of 10 times. This is a workaround to try and fix it.
    while True:
        try:
            # Test for "Failed to set power state" Error
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
            pipeline.start(config)
            # Test for RuntimeError: Frame didn't arrive within 5000
            frames = pipeline.wait_for_frames()
            _ = frames.get_color_frame().get_data()
            return pipeline
        except (RuntimeError, ConnectionError) as e:
            print(f"[MAC REALSENSE ERROR]: Trying to restart pipeline. Got: {e}")
            time.sleep(1)
            os.execv(sys.executable, ['python'] + sys.argv)

class LRUDict(OrderedDict):
    def __init__(self, maxsize=1000):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self: self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize: self.popitem(last=False)

class RealSenseConnector:
    def __init__(self):
        self._depth_cache = LRUDict(maxsize=3)
        self.pipeline = get_pipeline()

    def get_only_color_image(self) -> Tuple[np.ndarray, float]:
        frames = self.pipeline.wait_for_frames()
        timestamp = time.time()
        color_frame = frames.get_color_frame().get_data()
        return np.asanyarray(color_frame), timestamp

    def get_color_image(self) -> Tuple[np.ndarray, float]:
        frames = self.pipeline.wait_for_frames()
        timestamp = time.time()
        color_frame = frames.get_color_frame().get_data()
        depth_frame = frames.get_depth_frame().get_data()
        self._depth_cache[timestamp] = np.asanyarray(depth_frame)
        return np.asanyarray(color_frame), timestamp

    def get_depth_for_detection(self, detection: Detection) -> float:
        assert detection.timestamp is not None
        depth_array = self._depth_cache.get(detection.timestamp)
        if depth_array is None: return -1
        x, y, _ = map(int, detection.center_point)
        direct_distance = depth_array[y, x]
        x_norm = (x - c_x) / f_x
        y_norm = (y - c_y) / f_y
        z = direct_distance / np.sqrt(1 + x_norm**2 + y_norm**2)
        return z

if __name__ == "__main__":
    connector = RealSenseConnector()
    while True:
        color_image, timestamp = connector.get_color_image()