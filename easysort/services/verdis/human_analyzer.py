import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

VERBOSE: bool = False


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


def _normalize_label(text: str) -> str:
    """Map various human/AI labels to canonical tokens for comparison."""
    t = (text or "").strip().lower()
    if not t:
        return "unknown"
    if "stuck" in t:
        return "running_stuck"
    if "not running" in t and "belt" in t:
        return "not_running_belt_full"
    if t.startswith("not running"):
        return "not_running"
    if "running empty" in t:
        return "running_empty"
    if "plastic" in t:
        return "running_plastics"
    if "cardboard" in t:
        return "running_cardboard"
    if "other fraction" in t or "residual" in t or "mix" in t:
        return "running_other_fraction"
    return t


def _derive_group_key_from_clip(clip_path: Path) -> Tuple[str, str]:
    """Return (group_key, stem) for the clip to match analyzer outputs.
    Analyzer groups images by filename stem without the last underscore part.
    We'll try both the group key and full stem as possible json basenames.
    """
    stem = clip_path.stem
    # If the stem ends with _NN where NN are digits (image frame index), strip it to get the group key.
    # Otherwise, keep the full stem (e.g., HH_MM_SS for clips and JSONs).
    if re.search(r"_\d{2}$", stem):
        key = stem.rsplit("_", 1)[0]
    else:
        key = stem
    return key, stem


def _load_ai_category(images_dir: Path, clip_path: Path) -> Optional[str]:
    """Load AI category string for a clip by looking up <key>.json under images_dir."""
    key, stem = _derive_group_key_from_clip(clip_path)
    # Try the exact stem first (e.g., HH_MM_SS.json), then the group key fallback.
    candidates = [images_dir / f"{stem}.json", images_dir / f"{key}.json"]
    if VERBOSE:
        print(f"[_load_ai_category] clip={clip_path.name} stem={stem} key={key}")
        print(f"[_load_ai_category] images_dir={images_dir}")
        for cp in candidates:
            print(f"[_load_ai_category] try: {cp} exists={cp.exists()}")
    for p in candidates:
        if not p.exists():
            continue
        try:
            if VERBOSE:
                print(f"[_load_ai_category] opening: {p}")
            with open(p, "r") as f:
                data = json.load(f)
            cat = data.get("category")
            if VERBOSE:
                try:
                    keys_list = list(data.keys())
                except Exception:
                    keys_list = []
                print(f"[_load_ai_category] json keys={keys_list} category={cat}")
            if isinstance(cat, str) and cat.strip():
                return cat
            # fallback: sometimes content under raw or other keys
            raw = data.get("raw")
            if isinstance(raw, str):
                m = re.search(r'"category"\s*:\s*"([^"]+)"', raw)
                if m:
                    if VERBOSE:
                        print(f"[_load_ai_category] extracted category from raw: {m.group(1)}")
                    return m.group(1)
        except Exception as e:
            if VERBOSE:
                print(f"[_load_ai_category] error reading {p}: {e}")
            continue
    return None


def _compute_initial_stats(clips_dir: Path, images_dir: Path, label_store: LabelStore) -> Tuple[float, Dict[str, Dict[str, int]], int]:
    """Compute baseline accuracy and per-category true/false counts.
    Returns (accuracy, per_cat_stats, compared_count).
    per_cat_stats[human_cat] = {"true": x, "false": y}
    """
    clips = find_clips(clips_dir)
    root_dir = clips_dir.resolve()
    true_matches = 0
    compared = 0
    per_cat: Dict[str, Dict[str, int]] = {}

    for p in clips:
        rel = str(p.relative_to(root_dir).as_posix())
        if not label_store.is_labeled(rel):
            continue
        human_raw = label_store._labels.get(rel, "")
        ai_raw = _load_ai_category(images_dir, p)
        if not ai_raw:
            continue  # skip if no ai pred available
        h = _normalize_label(human_raw)
        a = _normalize_label(ai_raw)
        match = (h == a)
        compared += 1
        if match:
            true_matches += 1
        if h not in per_cat:
            per_cat[h] = {"true": 0, "false": 0}
        per_cat[h]["true" if match else "false"] += 1

    acc = (true_matches / compared) if compared > 0 else 0.0
    return acc, per_cat, compared


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


def evaluate_and_review(clips_dir: Path, images_dir: Optional[Path] = None) -> None:
    clips = find_clips(clips_dir)
    if len(clips) == 0:
        print("No clips found in", clips_dir)
        return

    root_dir = clips_dir.resolve()
    if images_dir is None:
        images_dir = root_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        print("images directory not found for analyzer outputs:", images_dir)
        return

    if VERBOSE:
        json_files = sorted([p for p in images_dir.glob("*.json")])
        jpg_files = sorted([p for p in images_dir.glob("*.jpg")])
        print(f"[evaluate] found JSONs={len(json_files)} JPGs={len(jpg_files)} in {images_dir}")

    label_store = LabelStore(root_dir)
    labeled_clips = [p for p in clips if label_store.is_labeled(str(p.relative_to(root_dir).as_posix()))]
    if len(labeled_clips) == 0:
        print("No labeled clips (labels.csv) found under", root_dir)
        return

    # Initial stats
    if VERBOSE:
        print(f"[evaluate] clips_dir={root_dir}")
        print(f"[evaluate] images_dir={images_dir}")
        print(f"[evaluate] total_clips={len(clips)} labeled={len(labeled_clips)}")
    acc, per_cat, compared = _compute_initial_stats(clips_dir, images_dir, label_store)
    print(f"Initial accuracy (AI vs human exact match): {acc:.3f} on {compared} pairs")
    print("Per-category (human label) true/false counts:")
    for cat in sorted(per_cat.keys()):
        tf = per_cat[cat]
        print(f"  {cat}: true={tf.get('true',0)} false={tf.get('false',0)}")
    if compared == 0:
        print("Note: 0 comparable pairs. Likely no matching AI JSONs were found for labeled clips.")
        print("Run with --verbose to see exact candidate JSON paths that are checked.")

    # Prepare UI
    window = "Eval Review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1800, 1100)
    side_w = 520

    # Prepare eval.csv
    eval_csv = root_dir / "eval.csv"
    write_header = not eval_csv.exists()
    eval_f = open(eval_csv, "a", newline="")
    eval_writer = csv.DictWriter(eval_f, fieldnames=[
        "clip", "human_label", "ai_label", "match", "user_mark", "timestamp"
    ])
    if write_header:
        eval_writer.writeheader()

    good_count = 0
    bad_count = 0
    reviewed = 0

    total = len(labeled_clips)
    for idx, clip_path in enumerate(labeled_clips, start=1):
        rel = str(clip_path.relative_to(root_dir).as_posix())
        human_raw = label_store._labels.get(rel, "")
        ai_raw = _load_ai_category(images_dir, clip_path) or "(missing)"
        if VERBOSE and ai_raw == "(missing)":
            print(f"[evaluate] AI prediction missing for {clip_path.name}")
        h = _normalize_label(human_raw)
        a = _normalize_label(ai_raw)
        eq = (h == a) and ai_raw != "(missing)"

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print("Warning: cannot open clip, skipping:", clip_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0.0:
            fps = 30.0
        delay = max(1, int(1000.0 / fps))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        user_mark: Optional[str] = None
        frames_read = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames_read = 0
                continue

            # Build left panel lines
            lines = [
                "Evaluation Review",
                f"Clip: {idx}/{total}",
                f"File: {clip_path.name}",
                f"Human: {human_raw}",
                f"AI: {ai_raw}",
                f"Match: {'YES' if eq else 'NO'}",
                "",
                f"Reviewed: {reviewed}  Good: {good_count}  Bad: {bad_count}",
                "",
                "Keys:",
                "  g = Mark good",
                "  b = Mark bad",
                "  n/SPACE/ENTER = Next",
                "  ESC = Quit",
            ]

            composite = compose_frame(frame, side_w, lines)

            # Draw a colored marker for match state (green/red) on the left panel area
            marker_color = (0, 200, 0) if eq else (0, 0, 200)
            cv2.circle(composite, (side_w - 30, 30), 14, marker_color, thickness=-1)

            cv2.imshow(window, composite)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyWindow(window)
                eval_f.close()
                return
            if key in (ord('g'), ord('G')):
                user_mark = "good"
                # continue showing until next is pressed
            elif key in (ord('b'), ord('B')):
                user_mark = "bad"
            elif key in (ord('n'), ord('N'), 13, 32):  # enter or space or n
                # finalize this clip
                ts = datetime.utcnow().isoformat()
                eval_writer.writerow({
                    "clip": rel,
                    "human_label": human_raw,
                    "ai_label": ai_raw,
                    "match": str(eq).lower(),
                    "user_mark": user_mark or "",
                    "timestamp": ts,
                })
                reviewed += 1
                if user_mark == "good":
                    good_count += 1
                elif user_mark == "bad":
                    bad_count += 1
                break

            frames_read += 1
            if frame_count > 0 and frames_read >= frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames_read = 0

        cap.release()

    # Finished review
    final_img = (np.zeros((600, 1000, 3), dtype=np.uint8))
    post_score = (good_count / reviewed) if reviewed > 0 else 0.0
    msg1 = f"Initial exact-match acc: {acc:.3f} on {compared} pairs"
    msg2 = f"Reviewed {reviewed} • Good: {good_count} • Bad: {bad_count}"
    msg3 = f"User good-rate: {post_score:.3f}"
    cv2.putText(final_img, msg1, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(final_img, msg2, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(final_img, msg3, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 200), 2, cv2.LINE_AA)
    cv2.imshow(window, final_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Human labeling and evaluation tool for 2s clips.")
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing 2s .mp4 clips")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode: compare with analyzer outputs and review")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to images dir containing analyzer JSONs (defaults to <clips_dir>/images)")
    parser.add_argument("--verbose", action="store_true", help="Verbose debug output for file matching and evaluation")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise SystemExit(f"clips_dir not found or not a directory: {clips_dir}")

    global VERBOSE
    VERBOSE = bool(args.verbose)

    if args.eval:
        images_dir = Path(args.images_dir) if args.images_dir else None
        evaluate_and_review(clips_dir, images_dir)
    else:
        play_and_label(clips_dir)


if __name__ == "__main__":
    main()


