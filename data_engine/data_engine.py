"""
Data Engine classes
- DataRecorder: Used to record and store new videos (POST)
- DataViewer:   Used to view video                  (GET)
- DataExplorer: Used to load, get and explore data  (GET)
"""

import subprocess
import os
import cv2
import time
import logging

def get_data_folder(): return os.path.join(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode('utf-8').strip("\n"), "data")

class DataRecorder:
    def __init__(self):
        self.data_folder = get_data_folder()
        if not os.path.exists(self.data_folder): os.makedirs(self.data_folder)

        self.recording = False
        self.out = None
        self.cap = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def quit(self): self.cap.release(); cv2.destroyAllWindows()
    def add_description(self, frame): 
        return cv2.rectangle(frame, (10, 10), (50, 30), (0, 0, 255), 2) if self.recording else cv2.rectangle(frame, (10, 10), (50, 30), (128, 128, 128), 2)

    def record(self, start): 
        if not start and self.recording: self.out.release(); self.recording = False; self.out = None
        if not self.recording and start: 
            self.out = cv2.VideoWriter(os.path.join(self.data_folder, f'recorded_video{int(time.time())}.mp4'), self.fourcc, 30.0, (1280, 720))
            self.recording = True

    def run(self):
        while True:
            _, frame = self.cap.read()
            cv2.imshow("Camera", self.add_description(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.quit(); break
            if key == ord('r'): self.record(start=True)
            if key == ord('s'): self.record(start=False)

class DataViewer:
    def __init__(self):
        self.data_folder = get_data_folder()
        self.files = os.listdir(self.data_folder)
        logging.info("\n".join(["You can choose from the following list: ", "--------", *self.files, "--------"]))

    def quit(self): self.cap.release(); cv2.destroyAllWindows()
    
    def view(self, index):
        if index > len(self.files) - 1: logging.warning(f"Your index is {index}, but max is {len(self.files) - 1}, so setting it to 0"); index = 0
        file_to_view = os.path.join(self.data_folder, self.files[index])
        logging.info(f"Viewing: {file_to_view}")
        self.cap = cv2.VideoCapture(file_to_view)
        while (self.cap.isOpened()): 
            _, frame = self.cap.read()
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): self.quit(); break

class DataExplorer:
    """
    
    """

if __name__ == "__main__":
    # DataRecorder().run()
    DataViewer().view(0)
