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
import numpy as np

from utils import get_free_filename

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s, %(levelname)s]: %(message)s')

def get_top_folder(): return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode('utf-8').strip("\n")

class BaseModel():
    def __init__(self, cap): 
        self.cap = cap; self.should_quit = False

    def quit(self):
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        self.should_quit = True
    
    def universal_keys(self, key):
        if key == ord('q'): self.quit()

    # All Editors has to have a .run with a folder parameter

class DataRecorder(BaseModel):
    def __init__(self):
        self.top_folder = get_top_folder()
        if not os.path.exists(self.top_folder): os.makedirs(self.top_folder)

        self.fps = 10
        self.recording = False
        self.frames_index = 0
        self.cap = None # buffer for later when it is actually used in .run and .quit from BaseModel
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

    def run(self, folder):
        self.cap = cv2.VideoCapture(0)
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
    def __init__(self, explore: str = "new"):
        # self.check_new_folder() # Autospit every 100 frames
        if explore not in ["new", "verified", "labelled"]:
            raise LookupError(f"Data folder to explore has to be one of {["new", "verified", "labelled"]}, but is {explore}")
        self.fps = 10
        self.data_folder = os.path.join(get_top_folder(), "data", explore)
        self.pause = False
        self.files_skipped = [] # when reloading folder, remember already skipped files
        self.files = sorted(os.listdir(self.data_folder))
        self.keyframes = []
        logging.info("\n".join(["You can choose from the following list: ", "--------", 
                                *[f"{l}: {df}" for l, df in enumerate(self.files)], "--------"]))
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


    def add_description(self, frame, video_info : dict = {}):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1

        frame = frame.copy()
        cv2.rectangle(frame, (0, 0), (280, 270), (255, 255, 255), -1)

        text = "Press 'p' to (un)pause"
        cv2.putText(frame, text, (10, 20), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'q' to quit"
        cv2.putText(frame, text, (10, 40), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'a' to accept"
        cv2.putText(frame, text, (10, 60), font, font_scale, (0, 0, 0), thickness)

        text = "Press 's' to skip"
        cv2.putText(frame, text, (10, 80), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'd' to delete"
        cv2.putText(frame, text, (10, 100), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'r' to review"
        cv2.putText(frame, text, (10, 120), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'b' to back 1 frame"
        cv2.putText(frame, text, (10, 140), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'n' to next 1 frames"
        cv2.putText(frame, text, (10, 160), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'k' to delete prev. frame"
        cv2.putText(frame, text, (10, 180), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'l' to accept prev. frames"
        cv2.putText(frame, text, (10, 200), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'i' to add keyframe"
        cv2.putText(frame, text, (10, 220), font, font_scale, (0, 0, 0), thickness)

        text = "Press 'o' to delete keyframe"
        cv2.putText(frame, text, (10, 240), font, font_scale, (0, 0, 0), thickness)

        text = "Is keyframe: " + str(video_info.get("keyframe_bool", "Unknown"))
        cv2.putText(frame, text, (10, 260), font, font_scale, (0,0,255) if video_info.get("keyframe_bool", False) else (0,0,0), thickness)

        return frame

    def prompt_index(self): return int(input("What index to you want to see (put -1 if all): "))
    
    def accept(self, file_path):
        verified_dir = 'data/verified'
        if not os.path.exists(verified_dir):
            os.makedirs(verified_dir)
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(verified_dir, filename))

    def accept_prev_frames(self, file_to_view, frame_files, up_to_index):
        new_filename = get_free_filename(file_to_view.split("_")[-2])
        verified_dir = 'data/verified'
        if not os.path.exists(verified_dir):
            os.makedirs(verified_dir)
        filename = os.path.basename(new_filename)
        frames_files_to_move = frame_files[:up_to_index]

        new_dir = os.path.join(verified_dir, filename)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for file in frames_files_to_move:
            shutil.move(os.path.join(file_to_view, file), new_dir)

    def delete_files(self, file_to_view, frame_files, up_to_index):
        for frame in frame_files[:up_to_index]:
            os.remove(os.path.join(file_to_view, frame))

    def delete(self, file_path):
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

    def add_keyframe(self, frame_index, file_to_view):
        self.keyframes.append(frame_index)
        self.write_keyframes(file_to_view)
        self.read_keyframes(file_to_view)

    def delete_keyframe(self, frame_index, file_to_view):
        self.keyframes.remove(frame_index)
        self.write_keyframes(file_to_view)
        self.read_keyframes(file_to_view)

    def read_keyframes(self, file_to_view):
        keyframes_path = os.path.join(file_to_view, "keyframes.txt")
        if not os.path.exists(keyframes_path): return
        with open(keyframes_path, "r") as f:
            self.keyframes = [line.strip() for line in f.readlines()]

    def write_keyframes(self, file_to_view):
        keyframes_path = os.path.join(file_to_view, "keyframes.txt")
        with open(keyframes_path, "w") as f:
            for name in self.keyframes:
                f.write(name + "\n")

    def _view(self, index: int = None):
        file_to_view = os.path.join(self.data_folder, self.files[index])
        logging.info(f"Viewing: {file_to_view}")
        frame_files = sorted([frame for frame in os.listdir(file_to_view) if frame[-4:] == ".jpg"])
        frame_index = 0
        self.read_keyframes(file_to_view)

        while True:
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 800, 600)
            frame_path = os.path.join(file_to_view, frame_files[frame_index])
            frame = cv2.imread(frame_path)
            frame_info = {"keyframe_bool": frame_index in self.keyframes}
            cv2.imshow("frame", self.add_description(frame, frame_info))
            key = cv2.waitKey(1000//self.fps) & 0xFF
            if key == ord('q'): self.quit(); break
            if key == ord('a'): self.accept(file_to_view); self.files_skipped.append(self.files[index]); return
            if key == ord('s'): self.files_skipped.append(self.files[index]); return
            if key == ord('d'): self.delete(file_to_view); return
            if key == ord('r'): frame_index = 0; self.pause = False
            if key == ord('p'): self.pause = not self.pause
            if key == ord('b'): frame_index -= 1
            if key == ord('n'): frame_index += 1
            if key == ord('k'): self.delete_files(file_to_view, frame_files, frame_index); self.pause = False; frame_index = 0; return False
            if key == ord('l'): self.accept_prev_frames(file_to_view, frame_files, frame_index); return True
            if key == ord("i"): self.add_keyframe(frame_index, file_to_view)
            if key == ord("o"): self.delete_keyframe(frame_index, file_to_view)

            if not self.pause:
                frame_index += 1

            if frame_index >= len(frame_files): frame_index = len(frame_files) - 1
            if frame_index < 0: frame_index = 0

    def _reload(self):
        self.files = sorted(os.listdir(self.data_folder))
        return [file for file in self.files if file not in self.files_skipped]
    
    def view(self, index: int = None):
        index = self.prompt_index() if index is None else index
        if index > len(self.files) - 1: logging.warning(f"Your index is {index}, but max is {len(self.files) - 1}, so setting it to -1"); index = -1
        if index >= 0: self._view(index); return
        while True:
            if self.should_quit: break
            unseen_files = self._reload()
            if len(unseen_files) == 0: break
            for i in range(len(self.files)): self._view(i)

class KeyframeEditor:
    def __init__(self): pass
    def run(self, folder): return

class Splitter:
    def __init__(self): pass
    def run(self, folder): return

class FrameEditor:
    def __init__(self): pass
    def run(self, folder): return

class LabelRunner:
    def __init__(self): pass
    def run(self, folder): return

class DataEngine():
    def __init__(self):
        self.dataMenu = DataMenu()
        self.dataRecorder = DataRecorder()
        self.keyframeEditor = KeyframeEditor()
        self.splitter = Splitter()
        self.frameEditor = FrameEditor()
        self.labelRunner = LabelRunner()
        self.run()

    def run(self):
        editor_type_to_editor_object = {
            "Recorder": self.dataRecorder.run,
            "Keyframe Editor": self.keyframeEditor.run,
            "Splitter": self.splitter.run,
            "Frame Editor": self.frameEditor.run,
            "Label Runner": self.labelRunner.run
        }
        while True:
            self.dataMenu.options_menu()
            editor = editor_type_to_editor_object[self.dataMenu.editor_type]
            editor(folder = self.dataMenu.folder_to_explore)


class DataMenu():
    def __init__(self):
        self.folder_to_explore = None
        self.editor_type = None
    
    def setup_menu(self):
        cv2.namedWindow("Options Menu", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Options Menu", 640, 480)

        # Initialize menu options
        self.folders = ["new", "verified", "labelled"]
        self.editors_without_folder = ["Recorder"]
        self.editors_where_folder_is_needed = ["Keyframe Editor", "Splitter", "Frame Editor", "Label Runner"]
        self.editors = self.editors_without_folder + self.editors_where_folder_is_needed
        self.confirmation_choices = ["Proceed"]
        self.selected_folder = 0
        self.selected_editor = 0
        self.selected_confirmation = 0
        self.global_buttons = ["back", "quit"] # add to last elif in each _logic if you add something here

        # If this is the first round at current menu setting:
        self.first_load_of_menu = True

        # when is what shown
        #   If you change any of these, make a _menu method and a _logic method and implement into options_menu()
        #   The last line of editor_logic also has to be changed
        #   _update_selected should also be updated
        self.current_step = 0  # 0: editor, 1: folder (if necessary), 2: confirm
        self.editor_menu_step = 0
        self.folder_menu_step = 1
        self.confirmation_menu_step = 2

        self.instructions = "Use W and S to navigate, and Enter to proceed."

    def _reset_appropriate_selected(self):
        if self.current_step == self.editor_menu_step: self.selected_editor = 0
        elif self.current_step == self.folder_menu_step: self.selected_folder = 0
        elif self.current_step == self.confirmation_menu_step: self.selected_confirmation = 0

    def folder_menu(self, img):
        for i, folder in enumerate(self.folders + self.global_buttons):
            text = folder if i != self.selected_folder else f"> {folder}"
            add_on = 1 if i > len(self.folders) - 1 else 0
            cv2.putText(img, text, (20, 50 + (i + add_on) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def folder_logic(self, key):
        if key == ord('w'): self.selected_folder = (self.selected_folder - 1) % (len(self.folders) + len(self.global_buttons))
        elif key == ord('s'): self.selected_folder = (self.selected_folder + 1) % (len(self.folders) + len(self.global_buttons))
        elif key == ord('\r'):
            self.first_load_of_menu = True
            back_selected = self.selected_folder == len(self.folders)
            quit_selected = self.selected_folder == len(self.folders) + 1

            if back_selected: self.current_step = max(self.current_step - 1, 0)
            elif quit_selected: cv2.destroyAllWindows(); exit()
            else: self.current_step += 1

    def editor_menu(self, img):
        for i, editor in enumerate(self.editors + self.global_buttons):
            text = editor if i != self.selected_editor else f"> {editor}"
            add_on = 1 if i > len(self.editors) - 1 else 0
            cv2.putText(img, text, (20, 50 + (i + add_on) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def editor_logic(self, key):
        if key == ord('w'): self.selected_editor = (self.selected_editor - 1) % (len(self.editors) + len(self.global_buttons))
        elif key == ord('s'): self.selected_editor = (self.selected_editor + 1) % (len(self.editors) + len(self.global_buttons))
        elif key == ord('\r'):
            self.first_load_of_menu = True
            back_selected = self.selected_editor == len(self.editors)
            quit_selected = self.selected_editor == len(self.editors) + 1

            if back_selected: self.current_step = max(self.current_step - 1, 0)
            elif quit_selected: cv2.destroyAllWindows(); exit()
            else: self.current_step += 1 if self.editors[self.selected_editor] in self.editors_where_folder_is_needed else 2

    def confirmation_menu(self, img):
        cv2.putText(img, f"Your choices", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Folder: {self.folders[self.selected_folder]}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Editor: {self.editors[self.selected_editor]}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, editor in enumerate(self.confirmation_choices + self.global_buttons):
            text = editor if i != self.selected_confirmation else f"> {editor}"
            add_on = 1 if i > len(self.editors) - 1 else 0
            cv2.putText(img, text, (20, 170 + (i + add_on) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def confirmation_logic(self, key):
        # Return True if the menu should close and proceed to the specified editor
        if key == ord('w'): self.selected_confirmation = (self.selected_confirmation - 1) % (len(self.confirmation_choices) + len(self.global_buttons))
        elif key == ord('s'): self.selected_confirmation = (self.selected_confirmation + 1) % (len(self.confirmation_choices) + len(self.global_buttons))
        elif key == ord('\r'):
            self.first_load_of_menu = True
            back_selected = self.selected_confirmation == len(self.confirmation_choices)
            quit_selected = self.selected_confirmation == len(self.confirmation_choices) + 1

            if back_selected: self.current_step = max(self.current_step - 1, 0) # if self.folder_to_explore
            elif quit_selected: cv2.destroyAllWindows(); exit()
            else:
                self.folder_to_explore = self.folders[self.selected_folder]
                self.editor_type = self.editors[self.selected_editor]
                print(f"Selected folder: {self.folder_to_explore}, Selected editor: {self.editor_type}")
                return True

    def options_menu(self):
        self.setup_menu()
        while True:
            if self.first_load_of_menu: self.first_load_of_menu = False; self._reset_appropriate_selected()

            # Create a black background
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(img, self.instructions, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.current_step == self.editor_menu_step: self.editor_menu(img)
            elif self.current_step == self.folder_menu_step: self.folder_menu(img)
            elif self.current_step == self.confirmation_menu_step: self.confirmation_menu(img)

            # Show and get user input
            cv2.imshow("Options Menu", img)
            key = cv2.waitKey(1) & 0xFF

            # Menu logic
            if self.current_step == self.editor_menu_step: self.editor_logic(key)
            elif self.current_step == self.folder_menu_step: self.folder_logic(key)
            elif self.current_step == self.confirmation_menu_step: 
                if self.confirmation_logic(key): break

        # Close the options menu window
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    DataEngine()
    # DataRecorder().run()
    # DataExplorer()#.view(-1)
    # DataExplorer(explore = "verified").view(-1)

# Todo:
# - Make into Menu that can do both recording and exploration/verification
#   - Make use of first_visit to reset index
#   - Skip folder if not necessary
# - Make Menu work with other classes
# - Make backbutton on the other classes