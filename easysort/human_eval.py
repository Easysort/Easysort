import cv2
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from easysort.registry import Registry
from easysort.helpers import REGISTRY_PATH, Sort
from easysort.sampler import Sampler
import datetime
from PIL import Image
from tqdm import tqdm

class HumanEvaluator:
    def __init__(self, paths: List[str], project: str, max_frames_per_video: int | None = None, sample_stride: int = 1, shuffle: bool = True):
        self.video_paths = paths
        self.project = project
        self.model = "HumanEval"
        self.max_frames_per_video = max_frames_per_video
        self.sample_stride = max(1, sample_stride)
        self.shuffle = shuffle

    def _full_video_path(self, registry_key: str) -> str:
        return str(Path(REGISTRY_PATH) / registry_key)

    def _label_key_base(self, video_key: str, frame_idx: int) -> str:
        # Base key used for both image and label for a specific frame
        return Registry.construct_path(video_key, self.model, self.project, f"annotations/{frame_idx:06d}")

    def _is_frame_labeled(self, video_key: str, frame_idx: int) -> bool:
        base = self._label_key_base(video_key, frame_idx)
        return Registry.EXISTS(f"{base}/label.json")

    def _candidate_frame_indices(self, video_key: str, total_frames: int) -> List[int]:
        indices = list(range(0, total_frames, self.sample_stride))
        # Remove frames already labeled
        indices = [i for i in indices if not self._is_frame_labeled(video_key, i)]
        if self.shuffle:
            random.shuffle(indices)
        if self.max_frames_per_video is not None:
            indices = indices[: self.max_frames_per_video]
        return indices

    def _save_annotation(self, video_key: str, frame_idx: int, frame: np.ndarray, label: int) -> None:
        base = self._label_key_base(video_key, frame_idx)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode frame to JPEG")
        # Ensure directory exists and save JPEG with .jpg extension
        out_dir = Path(REGISTRY_PATH) / base
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "image.jpg").write_bytes(buf.tobytes())
        # Save label metadata via registry
        Registry.POST(f"{base}/label", {"label": int(label), "frame_idx": int(frame_idx), "video": video_key})

    def _iter_frames(self, cap: cv2.VideoCapture, indices: List[int]) -> Tuple[int, np.ndarray]:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            yield idx, frame

    def run(self):
        window_name = "Human Evaluation - Space/0=class 0, 1-9=class 1-9, q=quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for video_key in self.video_paths:
            full_path = self._full_video_path(video_key)
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"Failed to open: {video_key}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = self._candidate_frame_indices(video_key, total_frames)
            if not indices:
                print(f"No unlabeled frames for: {video_key}")
                cap.release()
                continue

            for frame_idx, frame in self._iter_frames(cap, indices):
                # Double-check not labeled (race conditions)
                if self._is_frame_labeled(video_key, frame_idx):
                    continue
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord('q'), ord('Q')):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if key in (ord(' '), ord('0')):
                    label = 0
                elif ord('1') <= key <= ord('9'):
                    label = key - ord('0')
                else:
                    # Unrecognized key: skip without saving
                    continue
                self._save_annotation(video_key, frame_idx, frame, label)

            cap.release()

        cv2.destroyAllWindows()

def unpack_and_save(paths: List[str]):
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    for _ in tqdm(range(10)):
        path = random.choice(paths)
        frames = Sampler.unpack(path, crop="auto")
        for frame_idx, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f"{output_dir}/{path.split('/')[-1]}_{frame_idx}.jpg")



if __name__ == "__main__":
    # Registry.SYNC()
    paths = Registry.LIST(prefix="argo", suffix=".mp4")
    paths = Sort.since(paths, datetime.datetime(2025, 11, 17))
    paths = Sort.before(paths, datetime.datetime(2025, 11, 24))
    # random shuffle
    unpack_and_save(list(paths))
    # HumanEvaluator(paths, project="people_direction", max_frames_per_video=None, sample_stride=30, shuffle=True).run()
