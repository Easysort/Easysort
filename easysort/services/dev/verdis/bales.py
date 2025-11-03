from pathlib import Path
from typing import List, Tuple
import json

import cv2
import numpy as np


# Fixed crop for motion detection
CROP = {"x": 737, "y": 1043, "w": 681, "h": 473}


def sample_frames_every_second(video_path: Path, step_sec: int = 1) -> Tuple[List[np.ndarray], List[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0

    times: List[float] = []
    frames: List[np.ndarray] = []
    t = 0.0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
        times.append(t)
        t += step_sec
    cap.release()
    return frames, times


def compute_motion_flags(frames: List[np.ndarray], pix_delta: int = 10, thresh: float = 0.10) -> Tuple[List[float], List[bool]]:
    if len(frames) == 0:
        return [], []
    x, y, w, h = CROP["x"], CROP["y"], CROP["w"], CROP["h"]
    grays: List[np.ndarray] = []
    for fr in frames:
        H, W = fr.shape[:2]
        xx = max(0, min(W - 1, x)); yy = max(0, min(H - 1, y))
        ww = max(1, min(W - xx, w)); hh = max(1, min(H - yy, h))
        roi = fr[yy:yy+hh, xx:xx+ww]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.int16)
        grays.append(g)
    scores: List[float] = []
    flags: List[bool] = []
    prev: np.ndarray = grays[0]
    scores.append(0.0); flags.append(False)
    for g in grays[1:]:
        diff = np.abs(g - prev)
        frac = float((diff > pix_delta).sum()) / float(diff.size)
        scores.append(frac)
        flags.append(frac >= thresh)
        prev = g
    return scores, flags


def play_viewer(frames: List[np.ndarray], scores: List[float], flags: List[bool], fps: int = 15) -> None:
    window = "Bales Motion Player"
    delay = max(1, int(1000 / max(1, fps)))
    paused = False
    i = 0
    n = len(frames)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while 0 <= i < n:
        fr = frames[i].copy()
        H, W = fr.shape[:2]
        band_h = max(8, H // 60)
        color = (0, 200, 0) if flags[i] else (0, 0, 200)
        fr[0:band_h, 0:W] = color
        # draw crop rectangle for reference
        x, y, w, h = CROP["x"], CROP["y"], CROP["w"], CROP["h"]
        x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w)); h = max(1, min(H - y, h))
        cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 255), 2)
        txt = f"{i+1}/{n}  motion={scores[i]:.4f} ({'YES' if flags[i] else 'NO'})  Space=pause  n=next  b=back  ESC=quit"
        cv2.putText(fr, txt, (12, band_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.imshow(window, fr)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key == 27:  # ESC
            break
        if key in (ord(' '),):
            paused = not paused
            continue
        if key in (ord('b'), ord('B')):
            i = max(0, i - 1)
        elif key in (ord('n'), ord('N')):
            i = min(n - 1, i + 1)
        else:
            i = i + (0 if paused else 1)
    cv2.destroyAllWindows()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sample frames each second, detect motion on fixed crop, and review")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--step", type=int, default=1, help="Sampling step in seconds (default: 1)")
    parser.add_argument("--thresh", type=float, default=0.10, help="Motion threshold (fraction of changed pixels)")
    parser.add_argument("--pix_delta", type=int, default=10, help="Pixel intensity change threshold (0..255)")
    args = parser.parse_args()

    video_path = Path(args.video)
    frames, times = sample_frames_every_second(video_path, step_sec=int(args.step))
    print(f"Sampled {len(frames)} frames from {video_path}")
    scores, flags = compute_motion_flags(frames, pix_delta=int(args.pix_delta), thresh=float(args.thresh))
    # Optionally save
    out_path = video_path.with_suffix(".motion_seq.json")
    try:
        payload = [{"t": float(t), "score": float(s), "moving": bool(f)} for t, s, f in zip(times, scores, flags)]
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"Saved motion sequence to {out_path}")
    except Exception:
        pass
    play_viewer(frames, scores, flags)


if __name__ == "__main__":
    main()


