import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> None:
        self.x = max(0, min(self.x, max(0, width - 1)))
        self.y = max(0, min(self.y, max(0, height - 1)))
        self.w = max(1, min(self.w, max(1, width - self.x)))
        self.h = max(1, min(self.h, max(1, height - self.y)))

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def list_images(images_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._"):
            imgs.append(p)
    return imgs


def load_roi(path: Path) -> Optional[Roi]:
    return None


def choose_roi_with_mouse(image: np.ndarray, initial: Optional[Roi] = None, window: str = "Cropper - Select ROI") -> Optional[Roi]:
    disp = image.copy()
    if initial is not None:
        x, y, w, h = initial.as_tuple()
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1600, max(800, image.shape[1])), min(1200, max(600, image.shape[0])))
    rect = cv2.selectROI(window, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, w, h = map(int, rect)
    if w <= 0 or h <= 0:
        return None
    return Roi(x, y, w, h)


def draw_overlay(image: np.ndarray, roi: Optional[Roi]) -> np.ndarray:
    vis = image.copy()
    if roi is not None:
        x, y, w, h = roi.as_tuple()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def crop_image(image: np.ndarray, roi: Roi) -> np.ndarray:
    x, y, w, h = roi.as_tuple()
    return image[y : y + h, x : x + w]


def compose_preview(image: np.ndarray, roi: Optional[Roi]) -> np.ndarray:
    left = draw_overlay(image, roi)
    if roi is not None:
        right = crop_image(image, roi)
    else:
        right = np.zeros((image.shape[0], image.shape[1] // 2, 3), dtype=np.uint8)
    # scale both to same height for preview
    target_h = min(900, max(600, image.shape[0]))
    def _scale(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        scale = target_h / float(h)
        new_w = int(w * scale)
        return cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([_scale(left), _scale(right)])


def adjust_roi(roi: Roi, key: int, step: int, img_w: int, img_h: int) -> Roi:
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    # arrows: move; dedicated keys for resize to avoid platform issues
    # i/k = height -, height +; j/l = width -, width +
    if key == 81:  # left arrow
        x -= step
    elif key == 83:  # right arrow
        x += step
    elif key == 82:  # up arrow
        y -= step
    elif key == 84:  # down arrow
        y += step
    elif key in (ord('j'), ord('J')):
        w -= step
    elif key in (ord('l'), ord('L')):
        w += step
    elif key in (ord('i'), ord('I')):
        h -= step
    elif key in (ord('k'), ord('K')):
        h += step
    new = Roi(x, y, w, h)
    new.clamp(img_w, img_h)
    return new


def run(images_dir: Path, step: int = 5) -> None:
    imgs = list_images(images_dir)
    if len(imgs) == 0:
        raise SystemExit(f"No images found in {images_dir}")

    roi: Optional[Roi] = None

    idx = 0
    window = "Cropper"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        img_path = imgs[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            print("Warning: failed to read image, skipping:", img_path)
            idx = (idx + 1) % len(imgs)
            continue

        if roi is None:
            # Prompt selection on first load
            sel = choose_roi_with_mouse(image, None, window="Cropper - Select ROI")
            if sel is None:
                print("No ROI selected; press ESC to quit or 'r' to try again.")
            else:
                roi = sel

        if roi is not None:
            roi.clamp(image.shape[1], image.shape[0])

        preview = compose_preview(image, roi)
        # annotate
        info_lines = [
            f"Image {idx+1}/{len(imgs)}: {img_path.name}",
            "Keys: ←→↑↓ move • j/l width -/+ • i/k height -/+",
            "      r reselect • n next • p prev • q/ESC quit",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(preview, line, (20, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

        cv2.imshow(window, preview)
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC or q
            break
        if key in (ord('n'), ord('N')):
            idx = (idx + 1) % len(imgs)
        elif key in (ord('p'), ord('P')):
            idx = (idx - 1 + len(imgs)) % len(imgs)
        elif key in (ord('r'), ord('R')):
            sel = choose_roi_with_mouse(image, roi, window="Cropper - Reselect ROI")
            if sel is not None:
                roi = sel
        elif key != 255:  # any key pressed
            if roi is not None:
                roi = adjust_roi(roi, key, step, image.shape[1], image.shape[0])

    cv2.destroyAllWindows()
    # Print final ROI as JSON so user can hardcode later
    if roi is not None:
        print(json.dumps({"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h}))
    else:
        print("{}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive OpenCV cropper for selecting conveyor belt ROI.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing example images (e.g., <clips_dir>/images)")
    parser.add_argument("--step", type=int, default=5, help="Arrow/resize step in pixels")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"images_dir not found or not a directory: {images_dir}")

    run(images_dir, step=int(args.step))


if __name__ == "__main__":
    main()


