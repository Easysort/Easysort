import cv2 as cv
import time  # Import time for delay
import pyrealsense2 as rs
import numpy as np
from PyQt6.QtCore import (QThread, Qt, pyqtSignal)
from PyQt6.QtGui import (QPixmap, QImage)
from scipy.spatial.transform import Rotation as R
import math
import cv2


# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"

t_R_M = np.array([[0.0], [0.01], [0.49]])  # Example: Marker to Suction cup offset
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

        pipe_profile = self.pipeline.start(self.rs_config)
        depth_sensor = pipe_profile.get_device().first_depth_sensor()
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

            transform_translation_x = 0.0
            transform_translation_y = 0.0
            transform_translation_z = 0.0

            if marker_ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, .03, self.mtx, self.dst  # Now cameraMatrix and distCoeff are used
                )

                for i, marker_id in enumerate(marker_ids):

                    R_m, _ = cv2.Rodrigues(rvecs[i])
                    t_m = tvecs[i].reshape((3, 1))

                    # Construct Transformation Matrix from Camera to Marker
                    T_C_M = np.vstack((np.hstack((R_m, t_m)), [0, 0, 0, 1]))

                    # 623 is markerID which is mounted on the robot
                    marker_id = int(marker_ids[i][0])

                    if marker_id == 623:
                        T_C_R = np.dot(T_C_M, np.linalg.inv(T_R_M))
                        self.T_C_R = T_C_R
                        robot_position_in_camera = T_C_R[:3, 3]
                        self.robot_x = robot_position_in_camera[0]
                        self.robot_y = robot_position_in_camera[1]
                        self.robot_z = robot_position_in_camera[2]
                    elif marker_id == 163:
                        self.marker_positions[marker_id] = [
                            round(float(t_m[0]), 5),  # X position
                            round(float(t_m[1]), 5),  # Y position
                            round(float(t_m[2]), 5)  # Z position
                        ]
                    # Store the rotation information
                    #rotation_matrix = np.eye(4)
                    #rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                    #r = R.from_matrix(rotation_matrix[0:3, 0:3])
                    #quat = r.as_quat()

                    # Quaternion format
                    #transform_rotation_x = quat[0]
                    #transform_rotation_y = quat[1]
                    #transform_rotation_z = quat[2]
                    #transform_rotation_w = quat[3]

                    # Euler angle format in radians
                    #roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                    #                                               transform_rotation_y,
                    #                                               transform_rotation_z,
                    #                                               transform_rotation_w)

                    #roll_x = math.degrees(roll_x)
                    #pitch_y = math.degrees(pitch_y)
                    #yaw_z = math.degrees(yaw_z)

                    # Draw the axes on the marker

                    for i in range(len(rvecs)):
                        # Convert original rotation vector to rotation matrix
                        R_orig, _ = cv2.Rodrigues(rvecs[i])

                        # Define a 90° rotation about the Z-axis
                        theta = np.deg2rad(90)
                        R_z = np.array([
                            [np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]
                        ])

                        # Apply the additional rotation: new_R = R_z * R_orig
                        R_new = np.dot(R_z, R_orig)

                        # Convert the new rotation matrix back to a rotation vector
                        rvec_new, _ = cv2.Rodrigues(R_new)

                        # Draw the axes using the new rotation vector and original translation vector
                        cv2.drawFrameAxes(color_image, self.mtx, self.dst, rvec_new, tvecs[i], 0.05)

                    #cv2.drawFrameAxes(color_image, self.mtx, self.dst, rvecs[i], tvecs[i], 0.05)

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


