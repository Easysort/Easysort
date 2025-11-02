import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


Point = Tuple[int, int]


def draw_poly(img: np.ndarray, pts: List[Point], highlight: bool = True) -> np.ndarray:
    vis = img.copy()
    if len(pts) > 0:
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(vis, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], False, (0, 200, 0), 2)
        if highlight and len(pts) >= 3:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 180, 0))
            vis = cv2.addWeighted(overlay, 0.2, vis, 0.8, 0)
    return vis


def mask_from_polygon(size: Tuple[int, int], pts: List[Point]) -> np.ndarray:
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    if mask.ndim == 2:
        mask3 = cv2.merge([mask, mask, mask])
    else:
        mask3 = mask
    out[mask3 == 0] = 0
    return out


def interactive(image_path: Path, load_json: Optional[Path] = None) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    pts: List[Point] = []
    if load_json and load_json.exists():
        try:
            data = json.load(open(load_json, "r"))
            pts = [(int(p[0]), int(p[1])) for p in data.get("points", [])]
        except Exception:
            pts = []

    win = "Poly Crop"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.setMouseCallback(win, on_mouse)

    help_lines = [
        f"File: {image_path.name}",
        "Left-click: add point  •  Right-click: undo  •  c: clear  •  s: save/print  •  q/ESC: quit",
    ]
    while True:
        vis = draw_poly(img, pts)
        y = 26
        for line in help_lines:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            y += 26
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break
        if k in (ord('c'), ord('C')):
            pts.clear()
        if k in (ord('s'), ord('S')) and len(pts) >= 3:
            payload = {
                "image": str(image_path),
                "width": w,
                "height": h,
                "points": [(int(x), int(y)) for x, y in pts],
            }
            print(json.dumps(payload))
            try:
                with open(image_path.with_suffix(".poly.json"), "w") as f:
                    json.dump(payload, f)
            except Exception:
                pass
    cv2.destroyAllWindows()


def apply_to_image(image_path: Path, poly_json: Path, output: Optional[Path] = None) -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    data = json.load(open(poly_json, "r"))
    pts = [(int(p[0]), int(p[1])) for p in data.get("points", [])]
    mask = mask_from_polygon((img.shape[0], img.shape[1]), pts)
    out = apply_mask(img, mask)
    out_path = output or image_path.with_suffix(".poly.png")
    cv2.imwrite(str(out_path), out)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Polygon crop template tool (create/apply)")
    ap.add_argument("--image", type=str, required=True, help="Path to image")
    ap.add_argument("--load", type=str, default=None, help="Optional existing .poly.json to load")
    ap.add_argument("--apply", type=str, default=None, help="If set, apply this .poly.json to --image and save .poly.png")
    args = ap.parse_args()

    img_path = Path(args.image)
    if args.apply:
        out = apply_to_image(img_path, Path(args.apply))
        print(str(out))
    else:
        interactive(img_path, Path(args.load) if args.load else None)


if __name__ == "__main__":
    main()


