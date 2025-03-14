import cv2 as cv
import time  # Import time for delay
import pyrealsense2 as rs
import numpy as np
from PyQt6.QtCore import (QThread, Qt, pyqtSignal, )
from PyQt6.QtGui import (QPixmap, QImage)
import cv2

class Camera(QThread):
    emitImages = pyqtSignal(QImage)

    def __init__(self, config):
        super().__init__()
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, config.get('Camera', 'rgb_res_x'),
                                                         config.get('Camera', 'rgb_res_y'),
                                                         rs.format.bgr8,
                                                         config.get('Camera', 'max_fps'))

        self.rs_config.enable_stream(rs.stream.depth, config.get('Camera', 'depth_res_x'),
                                                         config.get('Camera', 'depth_res_y'),
                                                         rs.format.z16,
                                                         config.get('Camera', 'max_fps'))
        self.rs_frames = None
        self.rgb_frame = None
        self.depth_frame = None
        self.camera_run = True

        pipe_profile = self.pipeline.start(config)
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)

        s = pipe_profile.query_sensors()[1]
        s.set_option(rs.option.enable_auto_exposure, True)

    def stream_frames(self):
        while self.camera_run:
            try:
                self.rs_frames = self.pipeline.wait_for_frames()
                # Prevent from freezing
                self.rs_frames.keep()

                self.rgb_frame = self.rs_frames.get_color_frame()
                self.depth_frame = self.rs_frames.get_depth_frame()
                if not self.rgb_frame:
                    continue

            except Exception as e:
                print("Camera error: {}".format(e))
                break

            color_image = np.asanyarray(self.rgb_frame.get_data())
            # percent of original size
            width = int(color_image.shape[1])
            height = int(color_image.shape[0])
            dim = (width, height)
            color_image = cv2.resize(color_image, dim, interpolation=cv2.INTER_AREA)

            h, w, ch = color_image.shape

            bytesPerLine = ch * w
            convertToQtFormat = QImage(color_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(w, h, Qt.KeepAspectRatio)
            self.emitImages.emit(p)

    def get_image(self):
        if self.cam.isOpened():
            for _ in range(5):  # Read a few frames to flush buffer
                result, image = self.cam.read()
                time.sleep(0.05)  # Small delay
            if result:
                return image
        else:
            print("Camera is not ready!")


    def save_images(self, number):
        if self.cam.isOpened():
            print("Warming up the camera...")
            for _ in range(5):  # Discard the first few frames
                self.cam.read()
                time.sleep(0.05)

            for i in range(1, number + 1):
                input("Press Enter to capture")
                image = self.get_image()
                if image is not None:
                    cv.imwrite(f'/home/erikkocky/easySort/assets/{i}.png', image)


                time.sleep(0.1)
        else:
            print("Camera is not ready!")


