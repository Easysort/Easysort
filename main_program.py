import sys
from PyQt6 import QtCore, QtWidgets, QtGui, uic
from PyQt6.QtCore import QThreadPool, QRunnable, QTimer, QDateTime, QSettings, Qt, QUrl, QRect, pyqtSignal
from PyQt6.QtGui import (QPixmap, QImage, QTransform)
from easysort.system.camera import CameraClass
import configparser

# read config file
config = configparser.ConfigParser()
config.read('config.ini')

# startup camera
camera = CameraClass.Camera(config)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the UI file
        uic.loadUi("easysort_gui.ui", self)

        # Set the logo
        self.set_logo("assets/logo.png")

        # Connect button signals if needed
        self.pushButton1.clicked.connect(self.on_button1_clicked)
        self.pushButton2.clicked.connect(self.on_button2_clicked)
        self.pushButton3.clicked.connect(self.on_button3_clicked)

        self.camera.emitImages.connect(lambda p: self.setImage(p))
        self.camera.start()
        self.camImage1 = self.findChild(QtWidgets.QLabel, 'labelCamera')

        # Update status label, etc.
        self.label1.setText("Status: Ready")
        self.labelStatus.setText(config.get('Robot', 'Status'))

    def on_button1_clicked(self):
        self.label1.setText("Button 1 clicked")

    def on_button2_clicked(self):
        self.label1.setText("Button 2 clicked")

    def on_button3_clicked(self):
        self.label1.setText("Button 3 clicked")


    def setImage(self, p):
        p = QPixmap.fromImage(p)
        p = p.scaled(int(config.get('Camera', 'rgb_resolution_x')),
                     int(config.get('Camera', 'rgb_resolution_y')),
                     Qt.KeepAspectRatio)
        self.camImage1.setImage(p)


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
