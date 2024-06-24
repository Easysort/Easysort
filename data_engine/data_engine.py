"""
Data Engine classes
- DataRecorder: Used to record and store new videos         (POST)
- DataExplorer: Used to load, get, sort and explore data    (GET)
"""

import subprocess
import os
import cv2
import time
import logging
from datetime import datetime

from utils import get_free_filename

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s, %(levelname)s]: %(message)s')

def get_top_folder(): return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode('utf-8').strip("\n")

class BaseModel():
    def quit(self): self.cap.release(); cv2.destroyAllWindows()

class DataRecorder(BaseModel):
    def __init__(self):
        self.top_folder = get_top_folder()
        if not os.path.exists(self.top_folder): os.makedirs(self.top_folder)

        self.fps = 10
        self.recording = False
        self.cap = cv2.VideoCapture(0)
        self.frames_index = 0
        # self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        super().__init__()

    def add_description(self, frame): 
        return cv2.rectangle(frame, (10, 10), (50, 30), (0, 0, 255), 2) if self.recording else cv2.rectangle(frame, (10, 10), (50, 30), (128, 128, 128), 2)

    def record(self, start): 
        if not start and self.recording: self.recording = False; self.frames_index = 0
        if not self.recording and start: 
            self.data_folder = get_free_filename()
            if not os.path.exists(self.data_folder): os.makedirs(self.data_folder)
            self.recording = True
            self.frame_dir = os.path.join(self.top_folder, self.data_folder)

    def record_frame(self, frame):
        cv2.imwrite(os.path.join(self.frame_dir, f"frame_{self.frames_index:06d}.jpg"), frame)
        self.frames_index += 1

    def run(self):
        while True:
            _, frame = self.cap.read()
            cv2.imshow("Camera", self.add_description(frame.copy()))
            if self.recording: 
                self.record_frame(frame)
            key = cv2.waitKey(1000//self.fps) & 0xFF
            if key == ord('q'): self.quit(); break
            if key == ord('r'): self.record(start=True)
            if key == ord('s'): self.record(start=False)

class DataExplorer(BaseModel):
    def __init__(self, iterative: bool = False):
        self.fps = 10
        self.data_folder = os.path.join(get_top_folder(), "data")
        self.files = os.listdir(self.data_folder)
        logging.info("\n".join(["You can choose from the following list: ", "--------", 
                                *[f"{l}: {df}" for l, df in enumerate(self.files)], "--------"]))
        if iterative:
            self.iterative_index = int(input("What index to you want to see: "))
        super().__init__()
    
    def view(self, index: int = None):
        index = self.iterative_index if index is None else index
        if index is None: raise LookupError("No index was chosen. Set iterative = True when initializing or specify an index when running .view")
        if index > len(self.files) - 1: logging.warning(f"Your index is {index}, but max is {len(self.files) - 1}, so setting it to 0"); index = 0
        file_to_view = os.path.join(self.data_folder, self.files[index])
        logging.info(f"Viewing: {file_to_view}")
        frame_files = sorted(os.listdir(file_to_view))
        print(frame_files)
        for frame_file in frame_files:
            frame_path = os.path.join(file_to_view, frame_file)
            frame = cv2.imread(frame_path)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1000//self.fps) & 0xFF == ord('q'): self.quit(); break

if __name__ == "__main__":
    # DataRecorder().run()
    DataExplorer(iterative = True).view()
