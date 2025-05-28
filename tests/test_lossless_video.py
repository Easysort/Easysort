import numpy as np
from PIL import Image

from easysort.utils.lossless_video import read_mkv, save_lossless_mkv


def test_lossless_video(tmp_path):
    images = [Image.fromarray(np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8), "RGB") for _ in range(10)]

    save_lossless_mkv(images, tmp_path / "test_video.mkv", fps=1)

    read_images = read_mkv(tmp_path / "test_video.mkv")

    assert len(read_images) == len(images), "Number of frames read does not match number of frames written"
    for i, (read_image, original_image) in enumerate(zip(read_images, images)):
        read_array = np.array(read_image)
        original_array = np.array(original_image)
        assert read_array.shape == original_array.shape, (
            f"Frame {i} shape mismatch: {read_array.shape} != {original_array.shape}"
        )
        assert np.array_equal(read_array, original_array), f"Frame {i} does not match original image"
