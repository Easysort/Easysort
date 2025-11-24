
# Should take in a video, sample x frames with y seconds between them, and z seconds between groups
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from easysort.helpers import REGISTRY_PATH
from easysort.registry import Registry

@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int

ROSKILDE_CROP = Crop(x=0, y=0, w=2000, h=2000)
JYLLINGE_CROP = Crop(x=640, y=0, w=260, h=480)
DEVICE_TO_CROP = {"Argo-Jyllinge-Entrance-01": JYLLINGE_CROP, "Argo-roskilde-03-01": ROSKILDE_CROP}

class Sampler:
    @staticmethod
    def unpack(video_path: Path|str, crop: Crop|str = None) -> list[np.ndarray]:
        if isinstance(video_path, str): video_path = Path(video_path)
        if crop == "auto": crop = DEVICE_TO_CROP[video_path.parts[-6]]
        cap = cv2.VideoCapture(Registry._registry_path(str(video_path)))
        if not cap.isOpened(): raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps: raise RuntimeError("Could not determine FPS for video.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)  # (H, W, 3)
        cap.release()
        if len(frames) != total_frames: print(f"Warning: Expected {total_frames} frames, got {len(frames)}")
        if crop is not None: frames = [frame[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w] for frame in frames]
        return frames