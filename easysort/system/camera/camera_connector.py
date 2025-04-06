import cv2

class CameraConnector:
    def __init__(self):
        self.cap = cv2.VideoCapture(2)  # Use the default camera

    def get_color_image(self):
        ret, frame = self.cap.read()
        if not ret: raise RuntimeError("Failed to capture image from camera")
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        return frame, timestamp

    def get_depth_for_detection(self, detection):
        return 0

    def release(self):
        self.cap.release()
