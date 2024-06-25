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
import shutil

from utils import get_free_filename

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s, %(levelname)s]: %(message)s')

def get_top_folder(): return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode('utf-8').strip("\n")

class BaseModel():
    def __init__(self, cap): self.cap = cap
    def quit(self):
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

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
        super().__init__(self.cap)

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
        cv2.imwrite(os.path.join(self.frame_dir, f"frame_{self.frames_index:04d}.jpg"), frame)
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
    def __init__(self, data_folder_to_explore: str = "new"):
        if data_folder_to_explore not in ["new", "verified"]:
            raise LookupError(f"Data folder to explore has to be one of {["new", "verified"]}, but is {data_folder_to_explore}")
        self.fps = 10
        self.data_folder = os.path.join(get_top_folder(), "data", data_folder_to_explore)
        self.files = os.listdir(self.data_folder)
        logging.info("\n".join(["You can choose from the following list: ", "--------", 
                                *[f"{l}: {df}" for l, df in enumerate(self.files)], "--------"]))
        self.check_new_folder()
        super().__init__(None)

    def check_new_folder(self):
        print("Checking 'new' folder for videos too long...")
        path = os.path.join(get_top_folder(), "data", "new")
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            files = [f for f in os.listdir(folder_path) if f.startswith('frame_') and f.endswith('.jpg')]
            files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            num_parts = (len(files) + 99) // 100
            if num_parts < 2: continue
            print(num_parts, "for", folder_path)
            for i in range(num_parts):
                part_folder = os.path.join(path, f'{folder}_part_{i+1}')
                os.makedirs(part_folder, exist_ok=True)
                print("putting", part_folder)
                for j in range(100):
                    file_index = i * 100 + j
                    if file_index < len(files):
                        file_path = os.path.join(folder_path, files[file_index])
                        shutil.move(file_path, part_folder)
            self.delete(folder_path)
        print("Done checking")


    def add_description(self, frame):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1

        cv2.rectangle(frame, (0, 0), (200, 120), (255, 255, 255), -1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (200, 120), (0, 0, 0), -1)

        text = "Press 'q' to quit"
        cv2.putText(frame, text, (10, 20), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'a' to accept"
        cv2.putText(frame, text, (10, 40), font, font_scale, (0, 0, 0), thickness)

        text = "Press 's' to skip"
        cv2.putText(frame, text, (10, 60), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'd' to delete"
        cv2.putText(frame, text, (10, 80), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'r' to review"
        cv2.putText(frame, text, (10, 100), font, font_scale, (0, 0, 0), thickness)

        return frame

    def prompt_index(self): return int(input("What index to you want to see (put -1 if all): "))
    
    def accept(self, file_path):
        verified_dir = 'data/verified'
        if not os.path.exists(verified_dir):
            os.makedirs(verified_dir)
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(verified_dir, filename))

    def delete(self, file_path):
        # Operation not permitted: '/Users/lucasvilsen/Desktop/EasySort/data/new/d_2024-06-25_1'
        if os.path.exists(file_path):
            os.remove(file_path)

    def _view(self, index: int = None):
        file_to_view = os.path.join(self.data_folder, self.files[index])
        logging.info(f"Viewing: {file_to_view}")
        frame_files = sorted(os.listdir(file_to_view))
        for frame_file in frame_files:
            frame_path = os.path.join(file_to_view, frame_file)
            frame = cv2.imread(frame_path)
            cv2.imshow("frame", self.add_description(frame.copy()))
            key = cv2.waitKey(1000//self.fps) & 0xFF
            if key == ord('q'): self.quit(); break
            if key == ord('a'): self.accept(file_to_view); return
            if key == ord('s'): return
            if key == ord('d'): self.delete(file_to_view); return
            if key == ord('r'): self._view(index); return
        
        while True:
            cv2.imshow("frame", self.add_description(frame.copy()))
            key = cv2.waitKey(1000//self.fps) & 0xFF
            if key == ord('q'): self.quit(); break
            if key == ord('a'): self.accept(file_to_view); return
            if key == ord('s'): return
            if key == ord('d'): self.delete(file_to_view); return
            if key == ord('r'): self._view(index); return
    
    def view(self, index: int = None):
        index = self.prompt_index() if index is None else index
        if index > len(self.files) - 1: logging.warning(f"Your index is {index}, but max is {len(self.files) - 1}, so setting it to -1"); index = -1
        if index >= 0:
            self._view(index)
        else:
            for i in range(len(self.files)):
                self._view(i)


if __name__ == "__main__":
    # DataRecorder().run()
    DataExplorer().view(3)

# MISSING DELETE