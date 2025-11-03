import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def build_rotation_matrix(w: int, h: int, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    H = np.vstack([M, [0.0, 0.0, 1.0]])  # 3x3 homogeneous for later use
    return M, H


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive rotate+crop tool; prints rotation matrix and crop JSON for reuse")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--angle", type=float, default=0.0, help="Initial angle in degrees (positive = CCW)")
    args = parser.parse_args()

    img_path = Path(args.image)
    image = cv2.imread(str(img_path))
    if image is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    angle = float(args.angle)
    win = "Rectify - Rotate & Crop"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    crop: Optional[Tuple[int, int, int, int]] = None

    def overlay(im: np.ndarray) -> np.ndarray:
        out = im.copy()
        y = 26
        lines = [
            f"File: {img_path.name}",
            f"Angle: {angle:.2f} deg  [ ] adjust • r reset • c crop • s save/print • ESC quit",
        ]
        for line in lines:
            cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            y += 26
        if crop is not None:
            x, y0, w, h = crop
            cv2.rectangle(out, (x, y0), (x + w, y0 + h), (0, 200, 0), 2)
        return out

    while True:
        rotated = rotate_image(image, angle)
        view = overlay(rotated)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC
            break
        if k in (ord('['),):
            angle -= 0.5
        elif k in (ord(']'),):
            angle += 0.5
        elif k in (ord('r'), ord('R')):
            angle = 0.0
        elif k in (ord('c'), ord('C')):
            roi = cv2.selectROI(win, rotated, showCrosshair=True, fromCenter=False)
            x, y0, w, h = map(int, roi)
            crop = (x, y0, w, h) if w > 0 and h > 0 else None
        elif k in (ord('s'), ord('S')):
            h, w = image.shape[:2]
            M, H = build_rotation_matrix(w, h, angle)
            out = {
                "image": str(img_path),
                "angle_deg": round(angle, 6),
                "rotation_matrix_2x3": [[float(M[0, 0]), float(M[0, 1]), float(M[0, 2])],
                                         [float(M[1, 0]), float(M[1, 1]), float(M[1, 2])]],
                "rotation_matrix_3x3": [[float(H[0, 0]), float(H[0, 1]), float(H[0, 2])],
                                         [float(H[1, 0]), float(H[1, 1]), float(H[1, 2])],
                                         [float(H[2, 0]), float(H[2, 1]), float(H[2, 2])]],
                "crop": None if crop is None else {"x": crop[0], "y": crop[1], "w": crop[2], "h": crop[3]},
            }
            print(json.dumps(out))
            # try:
            #     with open(img_path.with_suffix(".rectify.json"), "w") as f:
            #         json.dump(out, f)
            # except Exception:
            #     pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


