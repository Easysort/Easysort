
from pathlib import Path
from typing import List
import json
from PIL import Image
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import base64

from easysort.services.dev.trainer.openai import OpenAITrainer, TrainerConfig
from easysort.services.verdis.runner import VerdisRunner
from easysort.services.dev.verdis.prompts import waste_type_belt_prompt, WasteTypeBeltJsonSchema


class VerdisBeltHandler:
    """
    Given a video path, this class will:
    - Split up into relevant sections
    - Detect the type of waste on the belt
    - Detect if the belt is moving or not
    - From those two classifications infer what the correct classification is
    - [Optionally] Be able to validate the classifications
    """
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.openai_trainer = OpenAITrainer()

    # Hardcoded polygon (belt region) provided by user
    # Coordinates are (x, y)
    POLY_POINTS = [
        (1418, 1512), (1019, 1502), (900, 704), (873, 76), (1000, 78), (1099, 747), (1406, 1510)
    ]

    @staticmethod
    def _poly_crop_to_b64(image_path: Path, points: List[tuple[int, int]]) -> str:
        im = cv2.imread(str(image_path))
        if im is None:
            return ""
        h, w = im.shape[:2]
        # Clamp points to image
        pts = np.array([[max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))] for x, y in points], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
        # Dim everything, then restore original inside polygon and draw a border to guide attention
        dim = (im * 0.25).astype(np.uint8)
        highlighted = dim.copy()
        highlighted[mask == 255] = im[mask == 255]
        cv2.polylines(highlighted, [pts], True, (0, 255, 255), 2)
        # Keep original dimensions to preserve scale/context
        ok, buf = cv2.imencode('.jpg', highlighted, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def detect_waste_type(self, group: List[Path]) -> str:
        # Use a single representative image (middle frame) to reduce cost
        if not group:
            return ""
        idx = max(0, min(len(group) // 2, len(group) - 1))
        # Apply polygon crop before sending to OpenAI
        img_b64 = self._poly_crop_to_b64(group[idx], self.POLY_POINTS)
        response = self.openai_trainer._openai_call(
            model="gpt-5-2025-08-07",
            prompt=waste_type_belt_prompt,
            images=[img_b64]
        )
        return WasteTypeBeltJsonSchema(**json.loads(response)).category
        

    def _belt_crop_image(self, image: Path) -> Image.Image:
        BELT_CROP = {"x": 862, "y": 45, "w": 332, "h": 1076}
        with Image.open(image) as im:
            im = im.convert("RGB")
            cropped = im.crop((BELT_CROP["x"], BELT_CROP["y"], BELT_CROP["x"] + BELT_CROP["w"], BELT_CROP["y"] + BELT_CROP["h"]))
            return cropped

    def detect_belt_moving(self, group: List[Path]) -> bool:
        cropped_images = [self._belt_crop_image(image) for image in group]
        if len(cropped_images) < 2:
            return False

        def to_gray_np(im: Image.Image) -> np.ndarray:
            g = im.convert("L")
            # downscale slightly for robustness/speed
            max_w = 512
            w, h = g.size
            if w > max_w:
                scale = max_w / float(w)
                g = g.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            arr = np.asarray(g, dtype=np.int16)
            return arr

        frames = [to_gray_np(im) for im in cropped_images]
        min_h = min(f.shape[0] for f in frames)
        min_w = min(f.shape[1] for f in frames)
        frames = [f[:min_h, :min_w] for f in frames]

        PIX_DELTA = 10  # intensity delta threshold (0..255)
        frac_changes: List[float] = []
        for a, b in zip(frames[:-1], frames[1:]):
            diff = np.abs(a - b)
            changed = (diff > PIX_DELTA).sum()
            frac = float(changed) / float(diff.size)
            frac_changes.append(frac)

        if not frac_changes:
            return False

        avg_change = float(np.mean(frac_changes))
        return avg_change

    def validate_classification(self, classification: str) -> bool:

        pass

    def run(self, concurrency: int = 8) -> None:
        groups = VerdisRunner(find_latest=False).analyze(self.video_path)
        print(f"Found {len(groups)} groups to process for {self.video_path}")
        results_path = self.video_path.with_suffix(".json")
        results_map = {}
        # Load existing results if any
        if results_path.exists():
            try:
                data = json.load(open(results_path, "r"))
                if isinstance(data, list):
                    results_map = {str(d.get("group_id")): d for d in data if isinstance(d, dict)}
                elif isinstance(data, dict):
                    # backward compatibility: single-entry format
                    gid = "__single__"
                    results_map[gid] = data
                print(f"Loaded existing detections for {len(results_map)} groups from {results_path}")
            except Exception:
                results_map = {}
                print(f"Warning: failed to read existing results at {results_path}; will recompute.")
        else:
            print(f"No previous detections found at {results_path}; starting fresh.")

        def group_id(paths: List[Path]) -> str:
            # derive from first image name: HH_MM_SS_00.jpg -> HH_MM_SS
            if not paths:
                return "unknown"
            stem = paths[0].stem
            return stem.rsplit("_", 1)[0] if "_" in stem else stem

        # Helper to get consistent group id
        # Compute motion for all groups first (local, fast)
        updated = False
        # Determine which groups need motion
        need_motion = []
        for grp in groups:
            gid = group_id(grp)
            if gid not in results_map or "motion" not in results_map[gid]:
                need_motion.append((gid, grp))
        print(f"Motion: {len(need_motion)} missing / {len(groups)} total")
        for gid, grp in tqdm(need_motion, desc="Motion detection"):
            motion_score = float(self.detect_belt_moving(grp))
            rec = results_map.get(gid, {"group_id": gid})
            rec["motion"] = motion_score
            results_map[gid] = rec
            updated = True

        # Then compute AI predictions in parallel for missing ones
        pending = []
        for grp in groups:
            gid = group_id(grp)
            rec = results_map.get(gid, {"group_id": gid})
            if "ai_category" not in rec or not rec["ai_category"]:
                pending.append((gid, grp))
        print(f"OpenAI: {len(pending)} missing / {len(groups)} total")
        if pending:
            with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
                futs = {ex.submit(self.detect_waste_type, grp): gid for gid, grp in pending}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="OpenAI predictions"):
                    gid = futs[fut]
                    try:
                        cat = fut.result()
                    except Exception as e:
                        cat = f"(error: {e})"
                    rec = results_map.get(gid, {"group_id": gid})
                    rec["ai_category"] = cat
                    results_map[gid] = rec
                    updated = True

        # Save if updated
        if updated or not results_path.exists():
            try:
                payload = [results_map[k] for k in sorted(results_map.keys())]
                with open(results_path, "w") as f:
                    json.dump(payload, f)
            except Exception:
                pass

        # Build a small timeline image of categories (top) + motion (bottom)
        def _build_timeline(base_groups: List[List[Path]], recs: dict, motion_thresh: float = 0.20) -> np.ndarray:
            n = len(base_groups)
            if n == 0:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            block_w = max(1, min(10, 800 // n if n > 0 else 4))
            W, H = max(1, block_w * n), 40
            img = np.zeros((H, W, 3), dtype=np.uint8)
            # BGR colors per category
            cat_colors = {
                "Cardboard": (19, 113, 199),
                "Paper": (220, 220, 220),
                "Residual": (192, 192, 128),
                "Plastics": (60, 160, 255),
                "Empty": (40, 40, 40),
            }
            for i, grp in enumerate(base_groups):
                x0, x1 = i * block_w, (i + 1) * block_w
                gid = group_id(grp)
                rec = recs.get(gid, {})
                cat = str(rec.get("ai_category", ""))
                color = cat_colors.get(cat, (90, 90, 90))
                img[0:20, x0:x1] = color
                m = float(rec.get("motion", 0.0))
                mcolor = (0, 200, 0) if m >= motion_thresh else (0, 0, 200)
                img[20:40, x0:x1] = mcolor
            return img

        timeline_base = _build_timeline(groups, results_map)
        # Show viewer for validation with navigation
        i = 0
        total = len(groups)
        while 0 <= i < total:
            grp = groups[i]
            gid = group_id(grp)
            rec = results_map.get(gid, {})
            ai = str(rec.get("ai_category", "(missing)"))
            mv = rec.get("motion", 0.0)
            action = self.view_group(grp, ai, mv, timeline_base, i, total)
            if action == "exit":
                break
            if action == "back":
                i = max(0, i - 1)
            else:
                i += 1
        

    def view_group(self, group: List[Path], ai_category: str, moving, timeline_base: np.ndarray = None, idx: int = 0, total: int = 1, window: str = "Verdis Belt Viewer") -> str:
        """Loop the 6 images with 0.5s delay, overlay AI/motion and a small timeline.
        'moving' can be a bool or a float motion score. The timeline shows per-group
        category (top) and motion (bottom), with current index highlighted.
        """
        if len(group) == 0:
            return "next"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        motion_score = float(moving) if not isinstance(moving, bool) else (1.0 if moving else 0.0)
        is_moving = motion_score >= 0.20
        info_lines = [f"AI: {ai_category}", f"Motion: {motion_score:.4f} ({'Moving' if is_moving else 'Still'})", "Press n or ESC to continue"]
        color = (0, 200, 0) if is_moving else (0, 0, 200)
        while True:
            for img_path in group:
                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                # draw timeline at top if provided
                if timeline_base is not None and timeline_base.size > 0:
                    Hf, Wf = im.shape[:2]
                    tl = cv2.resize(timeline_base, (Wf, 40), interpolation=cv2.INTER_NEAREST)
                    # highlight current position centered over the idx segment
                    x = int(((idx + 0.5) / max(1, total)) * tl.shape[1])
                    x = max(0, min(tl.shape[1] - 1, x))
                    cv2.line(tl, (x, 0), (x, tl.shape[0] - 1), (0, 0, 255), 2)
                    im = cv2.vconcat([tl, im])
                y = 28
                for line in info_lines:
                    cv2.putText(im, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
                    y += 32
                # motion indicator circle
                cv2.circle(im, (32, y + 10), 12, color, thickness=-1)
                cv2.imshow(window, im)
                k = cv2.waitKey(500) & 0xFF
                if k == 27:  # ESC exits all
                    cv2.destroyWindow(window)
                    return "exit"
                if k in (ord('b'), ord('B')):
                    cv2.destroyWindow(window)
                    return "back"
                if k in (ord('n'), ord('N')):
                    cv2.destroyWindow(window)
                    return "next"
        

if __name__ == "__main__":
    VerdisBeltHandler(Path("/Volumes/Easysort128/verdis/ARGO_ch5_20251028065738_20251028190253.mp4")).run()