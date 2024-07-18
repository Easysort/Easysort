
class DataExplorer(EditorBaseModel):
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
