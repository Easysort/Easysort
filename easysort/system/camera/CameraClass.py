import cv2 as cv
import time  # Import time for delay
import pyrealsense2 as rs
import numpy as np
from PyQt6.QtCore import (QThread, Qt, pyqtSignal)
from PyQt6.QtGui import (QPixmap, QImage)
from numpy.ma.extras import average
from scipy.spatial.transform import Rotation as R
import math
import cv2
from collections import deque

# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"

t_R_M = np.array([[0.0], [0.01], [0.19]])  # Example: Marker to Suction cup offset
R_R_M = np.eye(3)  # no rotation
T_R_M = np.vstack((np.hstack((R_R_M, t_R_M)), [0, 0, 0, 1]))

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.03

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class CameraClass(QThread):
    emitImages = pyqtSignal(QImage)
    robot_pose = pyqtSignal(list)
    markers_in_view = pyqtSignal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)  # Correct way to initialize QThread
        self.config = config  # Store config for later use
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.robot_x = 0
        self.robot_y = 0
        self.robot_z = 0
        self.T_C_M = None
        self.T_C_R = None
        self.marker_positions = {}
        self.pipe_profile = None
        self.object_position = [0,0,0]
        self.marker_history = []
        self.robot_history = []

        self.rs_config.enable_stream(rs.stream.color, int(config.get('Camera', 'rgb_res_x')),
                                                         int(config.get('Camera', 'rgb_res_y')),
                                                         rs.format.bgr8,
                                                         int(config.get('Camera', 'max_fps')))

        self.rs_config.enable_stream(rs.stream.depth, int(config.get('Camera', 'depth_res_x')),
                                                         int(config.get('Camera', 'depth_res_y')),
                                                         rs.format.z16,
                                                         int(config.get('Camera', 'max_fps')))
        self.rs_frames = None
        self.rgb_frame = None
        self.depth_frame = None
        self.camera_run = True

        self.pipe_profile = self.pipeline.start(self.rs_config)
        depth_sensor = self.pipe_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)


        self.mtx =  np.array([
                                [config.get('Camera', 'f_x'), 0., config.get('Camera', 'c_x')],
                                [0., config.get('Camera', 'f_y'), config.get('Camera', 'c_y')],
                                [0., 0., 1.]
                            ]).astype(float)
        self.dst = np.array([['-0.011645', '2.151450', '0.003982', '-0.006386', '-10.850711']]).astype(float)

        print("[INFO] detecting '{}' markers...".format(
            aruco_dictionary_name))
        self.this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
        self.this_aruco_parameters = cv2.aruco.DetectorParameters()


        #s = pipe_profile.query_sensors()[1]
        #s.set_option(rs.option.enable_auto_exposure, True)



    def run(self):
        while self.camera_run:
            try:
                self.rs_frames = self.pipeline.wait_for_frames()
                # This is preventing camera from occasional freezing
                self.rs_frames.keep()

                self.rgb_frame = self.rs_frames.get_color_frame()
                self.depth_frame = self.rs_frames.get_depth_frame()
                if not self.rgb_frame:
                    continue

            except Exception as e:
                print("Camera error: {}".format(e))
                break

            color_image = np.asanyarray(self.rgb_frame.get_data())

            corners, marker_ids, rejected = cv2.aruco.detectMarkers(
                color_image, self.this_aruco_dictionary, parameters=self.this_aruco_parameters
            )


            if marker_ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, .03, self.mtx, self.dst  # Now cameraMatrix and distCoeff are used
                )

                for i, marker_id in enumerate(marker_ids.flatten()):
                    marker_corners = corners[i][0]  # Get the four corners of the marker

                    # Get 3D position (X, Y, Z) in meters
                    marker_position = self.get_3d_position(self.depth_frame, marker_corners)

                    if marker_position is not None:
                        if marker_id == 163:
                            self.object_position = marker_position

                for i, marker_id in enumerate(marker_ids):

                    R_m, _ = cv2.Rodrigues(rvecs[i])
                    t_m = tvecs[i].reshape((3, 1))

                    # Construct Transformation Matrix from Camera to Marker
                    T_C_M = np.vstack((np.hstack((R_m, t_m)), [0, 0, 0, 1]))

                    marker_id = int(marker_ids[i][0])

                    if marker_id == 216:
                        T_C_R = np.dot(T_C_M, np.linalg.inv(T_R_M))
                        self.T_C_R = T_C_R
                        robot_position_in_camera = T_C_R[:3, 3]

                        self.robot_x = robot_position_in_camera[0] * 100
                        self.robot_y = robot_position_in_camera[1] * 100
                        self.robot_z = robot_position_in_camera[2] * 100


                    elif marker_id == 163:
                        self.marker_positions[marker_id] = [
                            round(float(self.object_position[0]), 5),  # X position
                            round(float(self.object_position[1]), 5),  # Y position
                            round(float(self.object_position[2]), 5)  # Z position
                        ]

                    cv2.drawFrameAxes(color_image, self.mtx, self.dst, rvecs[i], tvecs[i], 0.05)

            # percent of original size
            width = int(color_image.shape[1])
            height = int(color_image.shape[0])
            dim = (width, height)
            color_image = cv2.resize(color_image, dim, interpolation=cv2.INTER_AREA)

            h, w, ch = color_image.shape

            bytesPerLine = ch * w
            convertToQtFormat = QImage(color_image.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
            p = convertToQtFormat.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
            self.emitImages.emit(p)
            self.robot_pose.emit([self.robot_x,
                                  self.robot_y,
                                  self.robot_z])
            self.markers_in_view.emit(self.marker_positions)

    def get_image(self):
        if self.cam.isOpened():
            for _ in range(5):  # Read a few frames to flush buffer
                result, image = self.cam.read()
                time.sleep(0.05)  # Small delay
            if result:
                return image
        else:
            print("Camera is not ready!")

    def track_robot(self):
        pass

    def is_valid_movement(self, previous, current):
        threshold = 0.05
        if previous is None:
            return True  # No previous data, accept first measurement

        # Compute Euclidean distance between old and new positions
        distance = np.linalg.norm(np.array(previous) - np.array(current))

        return distance < threshold  # Return True if movement is within limits

    def get_3d_position(self, depth_frame, corners):
        """ Get the average 3D position from the ArUco marker corners. """
        depth_points = []
        intrinsics = self.pipe_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            depth = depth_frame.get_distance(x, y)  # Depth in meters

            if depth > 0:  # Valid depth check
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                depth_points.append(point_3d)

        if len(depth_points) > 0:
            return np.mean(depth_points, axis=0)  # Average position (X, Y, Z)

        return None  # No valid depth

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


