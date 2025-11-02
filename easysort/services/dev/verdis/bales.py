from pathlib import Path
from typing import List, Tuple
import json

import cv2
import numpy as np
from tqdm import tqdm

from easysort.services.verdis.runner import VerdisRunner


def group_id(paths: List[Path]) -> str:
    if not paths:
        return "unknown"
    stem = paths[0].stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def motion_score_for_group(img_paths: List[Path], pix_delta: int = 10, max_w: int = 768) -> float:
    frames: List[np.ndarray] = []
    for p in img_paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h, w = g.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(g.astype(np.int16))
    if len(frames) < 2:
        return 0.0
    min_h = min(f.shape[0] for f in frames)
    min_w = min(f.shape[1] for f in frames)
    frames = [f[:min_h, :min_w] for f in frames]
    fracs: List[float] = []
    for a, b in zip(frames[:-1], frames[1:]):
        diff = np.abs(a - b)
        fracs.append(float((diff > pix_delta).sum()) / float(diff.size))
    return float(np.mean(fracs)) if fracs else 0.0


def play_viewer(groups: List[List[Path]], scores: List[float], fps: int = 10, thresh: float = 0.20) -> None:
    window = "Motion Viewer"
    delay = max(1, int(1000 / max(1, fps)))
    paused = False
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    i = 0
    total = len(groups)
    while 0 <= i < total:
        gid = group_id(groups[i])
        score = scores[i] if i < len(scores) else 0.0
        color = (0, 200, 0) if score >= thresh else (0, 0, 200)
        for p in groups[i]:
            im = cv2.imread(str(p))
            if im is None:
                continue
            y = 28
            for line in [
                f"Group {i+1}/{total}  id={gid}",
                f"Motion score: {score:.4f}  ({'Moving' if score >= thresh else 'Still'})",
                "Space = pause/play • n = next • b = back • ESC = quit",
            ]:
                cv2.putText(im, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
                y += 26
            cv2.circle(im, (32, y + 10), 12, color, thickness=-1)
            cv2.imshow(window, im)
            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window)
                return
            if key in (ord(' '),):
                paused = not paused
                continue
            if key in (ord('b'), ord('B')):
                i = max(0, i - 1)
                break
            if key in (ord('n'), ord('N')):
                i = min(total - 1, i + 1)
                break
        else:
            i += 1
    cv2.destroyAllWindows()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Detect significant motion over time and review via OpenCV")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--thresh", type=float, default=0.10, help="Motion threshold for moving/still (default: 0.20)")
    parser.add_argument("--fps", type=int, default=10, help="Playback FPS in viewer (default: 10)")
    args = parser.parse_args()

    video_path = Path(args.video)
    groups = VerdisRunner(find_latest=False).analyze(video_path)
    print(f"Found {len(groups)} groups in {video_path}")

    scores: List[float] = []
    for grp in tqdm(groups, desc="Computing motion scores", unit="group"):
        scores.append(motion_score_for_group(grp))

    out_path = video_path.with_suffix(".motion.json")
    try:
        payload = [{"group_id": group_id(g), "motion": s} for g, s in zip(groups, scores)]
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"Saved motion scores to {out_path}")
    except Exception:
        pass

    play_viewer(groups, scores, fps=int(args.fps), thresh=float(args.thresh))


if __name__ == "__main__":
    main()


