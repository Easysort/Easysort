import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from easysort.utils.detections import Detection


def visualize_sorting_pipeline_image(
    image: np.ndarray,
    detections: List[Detection],
    *,
    show_plot: bool = True,
    shorten: bool = False,
    font_scale: float = 0.35,  # ⬑ smaller default
) -> np.ndarray:
    """
    Draw detections with a small, *stable* label anchored to box top‑left.

    Parameters
    ----------
    image       : BGR numpy image.
    detections  : list of your Detection objects.
    show_plot   : pop up matplotlib preview.
    shorten     : keep only the trailing part after the last ' - '.
    font_scale  : override font size (0.35 ≈ readable on 1080 p).
    """
    out = image.copy()
    H, W = out.shape[:2]  # pylint: disable=unused-variable

    def color_for(cls: str) -> tuple[int, int, int]:
        hue = hash(cls) % 179
        hsv = np.uint8([[[hue, 230, 230]]])  # type: ignore
        return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])  # type: ignore

    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        colour = color_for(d.class_name)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = d.class_name.split(" - ")[-1] if shorten else d.class_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # anchor just above top‑left; if that clips, drop it inside the box
        lx, by = x1, y1 - 4
        if by - th - 4 < 0:
            by = y1 + th + 4

        # opaque bar
        cv2.rectangle(out, (lx, by - th - 4), (lx + tw + 6, by + 2), colour, cv2.FILLED)

        # choose black/white text for contrast
        brightness = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
        txt_col = (0, 0, 0) if brightness > 127 else (255, 255, 255)

        cv2.putText(out, label, (lx + 3, by), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 1, cv2.LINE_AA)

    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return out
