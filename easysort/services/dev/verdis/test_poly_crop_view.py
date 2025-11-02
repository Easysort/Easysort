import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import base64

from easysort.services.dev.verdis.belt import VerdisBeltHandler


# Use the exact same polygon as the production code
POLY_POINTS: List[Tuple[int, int]] = VerdisBeltHandler.POLY_POINTS  # type: ignore


def poly_crop(image_path: Path, points: List[Tuple[int, int]]) -> np.ndarray:
    # Call the production helper used before sending to OpenAI
    b64 = VerdisBeltHandler._poly_crop_to_b64(image_path, points)  # type: ignore
    if not b64:
        raise SystemExit("poly crop returned empty image")
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    cropped = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if cropped is None:
        raise SystemExit("failed to decode cropped image")
    return cropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview hardcoded polygon crop for a given image")
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()
    p = Path(args.image)
    cropped = poly_crop(p, POLY_POINTS)
    im = cv2.imread(str(p))
    h, w = im.shape[:2]
    pts = np.array(POLY_POINTS, dtype=np.int32)
    overlay = im.copy()
    cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
    # Keep true dimensions side-by-side; right shows highlighted image (dimmed outside)
    vis = cv2.hconcat([overlay, cropped])
    cv2.namedWindow("PolyCrop Preview (left=overlay, right=cropped)", cv2.WINDOW_NORMAL)
    cv2.imshow("PolyCrop Preview (left=overlay, right=cropped)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


