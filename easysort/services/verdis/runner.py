import os
from pathlib import Path
from typing import Optional, Union
import cv2
import glob
import re
from datetime import datetime, timedelta
from tqdm import tqdm

class VerdisRunner:
    def __init__(self, folder: Optional[Union[str, Path]] = None, find_latest: bool = True):
        """
        Load the latest created video file in the given folder.
        """
        if not find_latest: self.video_path = None; return
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

    def analyze(self, video_path: Optional[Path] = None):
        """
        For every 2 minutes of video, save a 2 second clip to an output directory in the same place as the video.
        Additionally, extract 6 evenly-spaced images (JPEG) from each video.
        Handles large video files efficiently by not loading the entire video into memory.
        Returns: list of image groups, each group is a list of image Paths from one clip.
        """
        self.video_path = video_path if video_path is not None else self.video_path
        out_dir = Path(str(self.video_path.with_suffix("")) + "_clips") # add _clips 
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir = out_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Fast path: if images already exist, just return grouped paths without regenerating
        existing = sorted(images_dir.glob("*.jpg"))
        if existing:
            print(f"Found {len(existing)} existing images in {images_dir}")
            groups = []
            # group by stem before last underscore, e.g., HH_MM_SS_00.jpg -> HH_MM_SS
            tmp = {}
            for p in existing:
                stem = p.stem
                key = stem.rsplit("_", 1)[0] if "_" in stem else stem
                tmp.setdefault(key, []).append(p)
            for key in sorted(tmp.keys()):
                groups.append(sorted(tmp[key]))
            return groups

        assert self.video_path.exists() and self.video_path.is_file(), f"Video path {self.video_path} does not exist or is not a file."

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

        import math
        num_clips = math.ceil(duration_sec / two_min)

        start_dt: Optional[datetime] = None
        try:
            digits = re.findall(r"(\d{14})", self.video_path.stem)
            if digits:
                start_dt = datetime.strptime(digits[0], "%Y%m%d%H%M%S")
        except Exception:
            start_dt = None

        # Collect groups of image paths per clip
        groups = []
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
                    # Still append the group, but only existing files
                    groups.append(list(img_paths))
                    continue

                if curr_time >= duration_sec:
                    pbar.update(1)
                    # Still append empty/partial group if past end of video?
                    groups.append([p for p in img_paths if p.exists()])
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

                if need_clip and len(frames) == two_sec_frames:
                    height, width = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
                    for fr in frames:
                        writer.write(fr)
                    writer.release()

                # After (possibly) writing images, append the group paths
                groups.append(list(img_paths))

                pbar.update(1)

        cap.release()
        return groups

    def sample_images_every_minute(self, video_path: Optional[Path] = None):
        """
        Extract a single JPEG image for every minute of the video. No clips are generated.
        Images are saved next to the source video under a "<video_stem>_images" directory.
        Returns: list of image Paths.
        """
        self.video_path = video_path if video_path is not None else self.video_path
        assert self.video_path is not None, "video_path must be provided or set during initialization."
        assert self.video_path.exists() and self.video_path.is_file(), f"Video path {self.video_path} does not exist or is not a file."

        # Output directory: <video_stem>_images
        out_dir = Path(str(self.video_path.with_suffix("")) + "_images")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Fast path: if images already exist, just return paths without regenerating
        existing = sorted(out_dir.glob("*.jpg"))
        if existing:
            print(f"Found {len(existing)} existing images in {out_dir}")
            return existing

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps:
            raise RuntimeError("Could not determine FPS for video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        one_min = 60  # seconds

        import math
        num_minutes = max(1, math.ceil(duration_sec / one_min))

        # Try to infer a human-friendly timestamp base from the filename if present
        start_dt: Optional[datetime] = None
        try:
            digits = re.findall(r"(\d{14})", self.video_path.stem)
            if digits:
                start_dt = datetime.strptime(digits[0], "%Y%m%d%H%M%S")
        except Exception:
            start_dt = None

        saved_images = []
        with tqdm(total=num_minutes, desc="Extracting minute images", unit="img") as pbar:
            for idx in range(num_minutes):
                curr_time = idx * one_min
                if curr_time >= duration_sec:
                    pbar.update(1)
                    continue

                # Label as HH_MM_SS if timestamp base exists, otherwise zero-padded index
                if start_dt is not None:
                    ts_label = (start_dt + timedelta(seconds=curr_time)).strftime("%H_%M_%S")
                else:
                    ts_label = f"{idx:04d}"

                img_path = out_dir / f"{ts_label}.jpg"
                if img_path.exists():
                    saved_images.append(img_path)
                    pbar.update(1)
                    continue

                # Seek to the beginning of the minute window and capture one frame
                start_frame = int(curr_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(str(img_path), frame)
                    saved_images.append(img_path)
                pbar.update(1)

        cap.release()
        return saved_images

if __name__ == "__main__":
    runner = VerdisRunner()
    runner.analyze("/mnt/c/Users/lucas/Desktop/Verdis")