from pathlib import Path
from PIL import Image
from typing import List
import numpy as np

from easysort.runners.configs import VERDIS

class VerdisBelt:
    def __init__(self, video_path: str):
        self.video_path = video_path

    def _belt_crop_image(self, image: Image.Image) -> Image.Image:
        cropped = image.crop((VERDIS.BELT_CROP["x"], VERDIS.BELT_CROP["y"], VERDIS.BELT_CROP["x"] + VERDIS.BELT_CROP["w"], VERDIS.BELT_CROP["y"] + VERDIS.BELT_CROP["h"]))
        return cropped

    def sample_video(self) -> list[Image.Image]:
        pass

    def detect_belt_moving(self, group: List[Image.Image]) -> bool:
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