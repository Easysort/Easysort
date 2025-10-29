import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


LABEL_KEYS: Dict[str, str] = {
    "o": "Not running",
    "q": "Running empty",
    "w": "Running plastics",
    "e": "Running cardboard",
    "p": "Running stuck",
    "r": "Running other fraction",
    "i": "Not running, belt full",
}


def find_clips(clips_dir: Path) -> List[Path]:
    clips: List[Path] = []
    for p in sorted(clips_dir.glob("*.mp4")):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith("._") or name.startswith("."):
            continue
        try:
            if p.stat().st_size < 2048:
                continue
        except Exception:
            continue
        clips.append(p)
    return clips


class LabelStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.csv_path = self.root / "labels.csv"
        self._labels: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.csv_path.exists():
            try:
                with open(self.csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rel = row.get("clip") or ""
                        lab = row.get("label") or ""
                        if rel:
                            self._labels[rel] = lab
            except Exception:
                pass

    def save(self, rel_clip: str, label: str) -> None:
        ts = datetime.utcnow().isoformat()
        self._labels[rel_clip] = label
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip", "label", "timestamp"])
            if write_header:
                writer.writeheader()
            writer.writerow({"clip": rel_clip, "label": label, "timestamp": ts})

    def is_labeled(self, rel_clip: str) -> bool:
        return rel_clip in self._labels

    def labeled_count(self) -> int:
        return len(self._labels)


def draw_side_panel(height: int, width: int, lines: List[str]) -> cv2.Mat:
    panel = (np.zeros((height, width, 3), dtype=np.uint8))
    # Black already zero; ensure slight border on right
    x, y = 12, 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (200, 200, 200)
    line_h = 24
    for line in lines:
        cv2.putText(panel, line, (x, y), font, scale, color, 1, cv2.LINE_AA)
        y += line_h
    return panel


def compose_frame(frame: cv2.Mat, left_width: int, lines: List[str], target_height: Optional[int] = None) -> cv2.Mat:
    h, w = frame.shape[:2]
    if target_height is None:
        target_height = min(1080, max(720, h))
    scale = target_height / float(h)
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
    left = draw_side_panel(target_height, left_width, lines)
    composite = cv2.hconcat([left, resized])
    return composite


def play_and_label(clips_dir: Path) -> None:
    clips = find_clips(clips_dir)
    if len(clips) == 0:
        print("No clips found in", clips_dir)
        return

    root_dir = clips_dir.resolve()
    label_store = LabelStore(root_dir)

    total = len(clips)
    start_ts = time.time()

    # Seek first unlabeled index
    idx = 0
    for i, p in enumerate(clips):
        rel = str(p.relative_to(root_dir).as_posix())
        if not label_store.is_labeled(rel):
            idx = i
            break
    else:
        idx = total  # all done

    window = "Human Analyzer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1800, 1100)

    side_w = 480

    while idx < total:
        clip_path = clips[idx]
        rel = str(clip_path.relative_to(root_dir).as_posix())

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print("Warning: cannot open clip, skipping:", clip_path)
            idx += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0.0:
            fps = 30.0
        delay = max(1, int(1000.0 / fps))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Loop frames until user labels this clip
        loop = True
        go_back = False
        labeled_this = False
        frames_read = 0
        while loop:
            ok, frame = cap.read()
            if not ok or frame is None:
                # restart to loop the 2-second clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames_read = 0
                continue

            labeled = label_store.labeled_count()
            remaining = max(0, total - labeled)
            elapsed = max(0.0, time.time() - start_ts)
            avg_spc = (elapsed / labeled) if labeled > 0 else 0.0
            eta_sec = int(avg_spc * remaining) if labeled > 0 else 0
            eta_txt = str(timedelta(seconds=eta_sec)) if labeled > 0 else "—"

            lines = [
                "Human Analyzer",
                f"Clip: {idx+1}/{total}",
                f"File: {clip_path.name}",
                f"Labeled: {labeled} • Remaining: {remaining}",
                f"ETA: {eta_txt}",
                "",
                "Keys:",
                "  o = Not running",
                "  q = Running empty",
                "  w = Running plastics",
                "  e = Running cardboard",
                "  p = Running stuck",
                "  r = Running other fraction",
                "  i = Not running, belt full",
                "  b = Back to previous",
                "  ESC = Quit",
            ]

            composite = compose_frame(frame, side_w, lines)
            cv2.imshow(window, composite)

            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyWindow(window)
                return
            if key in (ord('b'), ord('B')):
                go_back = True
                loop = False
                break
            if key != 255:  # some key pressed
                ch = chr(key).lower()
                if ch in LABEL_KEYS:
                    label_store.save(rel, LABEL_KEYS[ch])
                    labeled_this = True
                    loop = False  # proceed to next clip
                    break
                # otherwise, ignore and continue
            # advance and loop at end
            frames_read += 1
            if frame_count > 0 and frames_read >= frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames_read = 0

        cap.release()
        if go_back:
            idx = max(0, idx - 1)
        elif labeled_this:
            idx += 1
        else:
            idx += 1

    # Finished all clips
    final_img = (np.zeros((600, 1000, 3), dtype=np.uint8))
    cv2.putText(final_img, "All clips labeled.", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 255, 200), 2, cv2.LINE_AA)
    cv2.imshow(window, final_img)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label 2-second clips with an OpenCV player.")
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing 2s .mp4 clips")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise SystemExit(f"clips_dir not found or not a directory: {clips_dir}")
    play_and_label(clips_dir)


if __name__ == "__main__":
    main()


