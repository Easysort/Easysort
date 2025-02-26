
import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm

from easysort.sorting.pipeline import SortingPipeline


def visualize_video(uuid: str, low: int = 0, high: int = 1000):
    pipeline = SortingPipeline()
    video_new_path = f"/Users/lucasvilsen/Documents/Documents/easylabeller/data/new/{uuid}"
    video_verified_path = f"/Users/lucasvilsen/Documents/Documents/easylabeller/data/verified/{uuid}"
    video_path = video_new_path if os.path.exists(video_new_path) else video_verified_path
    image_paths = sorted(glob.glob(f"{video_path}/*.jpg"))
    rendered_images_path = Path(f"/Users/lucasvilsen/Documents/Documents/EasySort/rendered_videos/{uuid}")
    rendered_images_path.mkdir(parents=True, exist_ok=True)
    for file in rendered_images_path.glob("*.jpg"): file.unlink()
    for file in rendered_images_path.glob("*.mp4"): file.unlink()

    rendered_images = []
    frame_number_to_image_path = {int(image_path.split("_")[-1].split(".")[0]): image_path for image_path in image_paths}
    image_paths = sorted([frame_number_to_image_path[i] for i in range(low, high)])
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        detections = pipeline(image)
        main_view = pipeline.visualize(image, detections, show_plot=False)
        rendered_images.append(main_view)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(rendered_images_path / f"{uuid}.mp4"), fourcc, 10, (rendered_images[0].shape[1], rendered_images[0].shape[0]))
    for image in rendered_images: video_writer.write(image)
    video_writer.release()


if __name__ == "__main__":
    visualize_video("d_2024-06-27_2", low=28, high=69)