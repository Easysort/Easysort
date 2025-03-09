import cv2 as cv

def start_streaming(port):
    try:
        return cv.VideoCapture(port)
    except Exception as e:
        print("[Camera init error]: {}".format(e))

class Camera:
    def __init__(self, port: int):
        self.port = port
        self.cam = start_streaming(self.port)

    def get_image(self):
        if self.cam.isOpened():
            result, image = self.cam.read()
            if result:
                return image
        else:
            print("Camera is not ready!")

