import sys
import threading

from PyQt6 import QtCore, QtWidgets, QtGui, uic
from PyQt6.QtCore import QThreadPool, QRunnable, QTimer, QDateTime, QSettings, Qt, QUrl, QRect, pyqtSignal
from PyQt6.QtGui import (QPixmap, QImage, QTransform)
from easysort.common.environment import Environment
import configparser
import numpy as np

from easysort.system.camera.CameraClass import CameraClass
from easysort.system.gantry.connector import GantryConnector
# read config file
config = configparser.ConfigParser()
config.read('config.ini')

# startup camera
camera = CameraClass(config=config)
robot = GantryConnector(Environment.GANTRY_PORT)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        uic.loadUi("easysort_gui.ui", self)

        # Set the logo
        self.set_logo("docs/Easysort_logo.png")

        # Connect button signals if needed
        self.pushButtonXplus.clicked.connect(self.robotXplus)
        self.pushButtonXminus.clicked.connect(self.robotXminus)
        self.pushButtonYplus.clicked.connect(self.robotYplus)
        self.pushButtonYminus.clicked.connect(self.robotYminus)
        self.pushButtonZplus.clicked.connect(self.robotZplus)
        self.pushButtonZminus.clicked.connect(self.robotZminus)
        self.suctionOnButton.clicked.connect(self.suctionOn)
        self.suctionOffButton.clicked.connect(self.suctionOff)
        self.pickUpButton.clicked.connect(self.pickUp)

        camera.emitImages.connect(lambda p: self.setImage(p))
        camera.robot_pose.connect(self.setRobotPose)
        camera.markers_in_view.connect(self.markersPositions)

        camera.start()
        self.camImage1 = self.findChild(QtWidgets.QLabel, 'labelCamera')

        # Update status label, etc.
        self.label1.setText("Status: Ready")
        self.labelStatus.setText(config.get('Robot', 'Status'))

    def robotXplus(self):
        self.label1.setText("Robot X+")
        t = threading.Thread(target=robot.go_to, args=(robot.position[0] + 1,robot.position[1], robot.position[2]))
        t.start()
        #self.move_robot_camera()

    def robotXminus(self):
        self.label1.setText("Robot X-")
        t = threading.Thread(target=robot.go_to, args=(robot.position[0] - 1 ,robot.position[1], robot.position[2]))
        t.start()

    def robotYplus(self):
        self.label1.setText("Robot Y+")
        t = threading.Thread(target=robot.go_to, args=(robot.position[0] ,robot.position[1] + 5, robot.position[2]))
        t.start()

    def robotYminus(self):
        self.label1.setText("Robot Y-")
        t = threading.Thread(target=robot.go_to, args=(robot.position[0] ,robot.position[1] - 5, robot.position[2]))
        t.start()

    def robotZplus(self):
        self.label1.setText("Robot Z+")
        print("Z+")
        t = threading.Thread(target=robot.go_to, args=(robot.position[0] ,robot.position[1], robot.position[2] + 2))
        t.start()

    def robotZminus(self):
        self.label1.setText("Robot Z-")
        t = threading.Thread(target=robot.go_to, args=(robot.position[0] , robot.position[1], robot.position[2] - 1))
        t.start()

    def suctionOn(self):
        t = threading.Thread(target=robot.suction_on)
        t.start()

    def suctionOff(self):
        t = threading.Thread(target=robot.suction_off)
        t.start()

    def setRobotPose(self, pose):
        self.labelStatus.setText(f"{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}")

    def markersPositions(self, marker_positions):
        for marker_id, position in marker_positions.items():
            self.markersLabel.setText(f"Marker {marker_id}:"
                                      f" X={position[0]:.3f},"
                                      f" Y={position[1]:.3f},"
                                      f" Z={position[2]:.3f}")

    def pickUp(self):
        self.label1.setText("Move robot to the marker")

        for marker_id, marker_pose in camera.marker_positions.items():
            print(marker_id)
            T_R_C = np.linalg.inv(camera.T_C_R)

            marker_cam = np.array([[marker_pose[0]], [marker_pose[1]], [marker_pose[2]], [1]])

            # Transform marker position from Camera Frame to Robot Frame
            marker_robot = np.dot(T_R_C, marker_cam)[:3]  # Extract only XYZ


            print("move robot to {}".format([robot.position[0] + float(marker_robot[1]),
                                             robot.position[1] + float(-marker_robot[0]),
                                             robot.position[2] + float(marker_robot[2])]))

            x = (robot.position[0] + float(-marker_robot[1])) * 100
            y = (robot.position[1] + float(marker_robot[0])) * 100
            z = (robot.position[2] + float(marker_robot[2])) * 100

        print(x,y,z)
        print("****")
        t = threading.Thread(target=robot.go_to, args=(x,y,z))
        t.start()

    def setImage(self, p):
        p = QPixmap.fromImage(p)
        p = p.scaled(int(config.get('Camera', 'rgb_res_x')),
                     int(config.get('Camera', 'rgb_res_y')),
                     Qt.AspectRatioMode.KeepAspectRatio)
        self.camImage1.setPixmap(p)


    def set_logo(self, filepath):
        pixmap = QtGui.QPixmap(filepath)
        if not pixmap.isNull():
            self.labelLogo.setPixmap(
                pixmap.scaled(
                    self.labelLogo.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            self.labelLogo.setText("Logo not found")

    def set_camera_image(self, filepath):
        pixmap = QtGui.QPixmap(filepath)
        if not pixmap.isNull():
            self.labelCamera.setPixmap(
                pixmap.scaled(
                    self.labelCamera.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            self.labelCamera.setText("Camera feed not available")



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
