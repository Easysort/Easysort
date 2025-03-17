from easysort.utils.detections import Detection
from typing import Tuple
import time
import numpy as np
import pyrealsense2 as rs
from functools import lru_cache

class RealSenseConnector:
    def __init__(self):
        self._get_depth_cached = lru_cache(maxsize=100)(self._get_depth)

    def setup(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        return self.pipeline

    def get_color_image(self) -> Tuple[np.ndarray, float]:
        frames = self.pipeline.wait_for_frames()
        self._last_timestamp = time.time()
        color_frame = frames.get_color_frame().get_data()
        depth_frame = frames.get_depth_frame().get_data()
        self._get_depth_cached(self._last_timestamp, np.asanyarray(depth_frame))
        return np.asanyarray(color_frame), self._last_timestamp

    def _get_depth(self, _: float, depth_array: np.ndarray) -> np.ndarray:
        return depth_array

    def get_depth_for_detection(self, detection: Detection) -> float: # TODO: Handle this better and consider pythagoras
        assert detection.timestamp is not None
        return self._get_depth_cached(detection.timestamp)[int(detection.center_point[1]), int(detection.center_point[0])] or -1