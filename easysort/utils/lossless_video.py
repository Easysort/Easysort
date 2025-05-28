from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


def save_lossless_mkv(images: list[Image.Image], path: Path, fps: int = 1) -> None:
    assert images, "Image list cannot be empty"
    assert all(image.mode == "RGB" for image in images), "All images must be in RGB mode"
    assert path.suffix == ".mkv", "Path must have .mkv extension"
    assert fps > 0, "FPS must be a positive integer"
    arrays = [np.asarray(image) for image in images]
    assert all(array.ndim == 3 and array.shape[2] == 3 for array in arrays), "All images must be RGB arrays"
    assert all(array.dtype == np.uint8 for array in arrays), "All images must be of type uint8"
    iio.imwrite(
        path,
        arrays,
        plugin="pyav",
        codec="ffv1",
        fps=fps,
        in_pixel_format="rgb24",
        out_pixel_format="bgr0",
    )


def read_mkv(path: Path) -> list[Image.Image]:
    read_images = iio.imread(path, plugin="pyav", format="rgb24")
    return [Image.fromarray(image) for image in read_images]
