
# Should take in a video, sample x frames with y seconds between them, and z seconds between groups
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int

class Sampler:
    @staticmethod
    def unpack(video_path: Path, crop: Crop = None) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps: raise RuntimeError("Could not determine FPS for video.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        pbar = tqdm(total=total_frames, desc="Unpacking video")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)  # (H, W, 3)
            pbar.update(1)
        cap.release()
        pbar.close()
        if len(frames) != total_frames: print(f"Warning: Expected {total_frames} frames, got {len(frames)}")
        if crop is not None:
            frames = [frame[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w] for frame in frames]
        return frames