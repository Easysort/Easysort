import os
from pathlib import Path
from typing import Optional, Union
import cv2
import glob
import re
from datetime import datetime, timedelta
from tqdm import tqdm

class VerdisRunner:
    def __init__(self, folder: Union[str, Path]):
        """
        Load the latest created video file in the given folder.
        """
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise ValueError(f"{self.folder} is not a valid directory.")
        video_files = sorted(
            self.folder.glob("*.mp4"), 
            key=lambda f: f.stat().st_ctime, 
            reverse=True
        )
        if not video_files:
            raise FileNotFoundError(f"No MP4 files found in {self.folder}")
        self.video_path = video_files[0]
        self.id = self.video_path.stem

    def analyze(self, name: Optional[str] = None) -> None:
        """
        For every 2 minutes of video, save a 2 second clip to an output directory in the same place as the video.
        Additionally, extract 6 evenly-spaced images (JPEG) from each video.
        Handles large video files efficiently by not loading the entire video into memory.
        """
        out_dir = self.video_path.parent / (name or (self.id + "_clips"))
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps:
            raise RuntimeError("Could not determine FPS for video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        two_min = 2 * 60          # 2 minutes in seconds
        two_sec = 2               # 2 seconds in seconds
        two_sec_frames = int(two_sec * fps)

        # Calculate the number of clips to extract
        # If curr_time=0, next is 120, next is 240... so num_clips = ceil(duration_sec / two_min)
        import math
        num_clips = math.ceil(duration_sec / two_min)

        # derive start datetime from filename if present (YYYYMMDDHHMMSS)
        start_dt: Optional[datetime] = None
        try:
            digits = re.findall(r"(\d{14})", self.video_path.stem)
            if digits:
                start_dt = datetime.strptime(digits[0], "%Y%m%d%H%M%S")
        except Exception:
            start_dt = None

        # For each 2-second clip window, optionally write the clip and extract 6 images inside that window
        num_images_per_clip = 6
        with tqdm(total=num_clips, desc="Processing clips & images", unit="clip") as pbar:
            for idx in range(num_clips):
                # compute timestamp label for this window
                curr_time = idx * two_min
                if start_dt is not None:
                    ts_label = (start_dt + timedelta(seconds=curr_time)).strftime("%H_%M_%S")
                else:
                    ts_label = f"{idx:04d}"

                # Targets based on timestamp
                clip_path = out_dir / f"{ts_label}.mp4"
                img_paths = [images_dir / f"{ts_label}_{i:02d}.jpg" for i in range(num_images_per_clip)]

                need_clip = not clip_path.exists()
                need_images = not all(p.exists() for p in img_paths)
                if not need_clip and not need_images:
                    pbar.update(1)
                    continue

                # Compute start frame for this 2s window and set capture
                if curr_time >= duration_sec:
                    pbar.update(1)
                    continue
                start_frame = int(curr_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                # Precompute image frame offsets within the 2-second window: 1/7..6/7 of two_sec_frames
                img_offsets = [max(0, min(two_sec_frames - 1, int((i+1) * two_sec_frames / (num_images_per_clip + 1))))
                               for i in range(num_images_per_clip)]
                img_offsets_set = set(img_offsets)

                frames = []
                wrote_images = [p.exists() for p in img_paths]
                for fidx in range(two_sec_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if need_clip:
                        frames.append(frame)
                    if need_images and fidx in img_offsets_set:
                        i = img_offsets.index(fidx)
                        if not wrote_images[i]:
                            cv2.imwrite(str(img_paths[i]), frame)
                            wrote_images[i] = True

                # write clip if we captured full 2s worth of frames
                if need_clip and len(frames) == two_sec_frames:
                    height, width = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
                    for fr in frames:
                        writer.write(fr)
                    writer.release()

                pbar.update(1)

        cap.release()

if __name__ == "__main__":
    runner = VerdisRunner("/Volumes/Easysort128/verdis")
    runner.analyze()