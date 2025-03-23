
import os
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

from easysort.sorting.pipeline import SortingPipeline
from easysort.visualize.sorting_image import visualize_sorting_pipeline_image

def visualize_video(uuid: str, low: int = 0, high: int = 1000, save_images: bool = False):
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
        main_view = visualize_sorting_pipeline_image(image, detections, show_plot=False)
        rendered_images.append(main_view)

    if save_images:
        for i, image in enumerate(rendered_images):
            cv2.imwrite(str(rendered_images_path / f"{i}.jpg"), image)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(rendered_images_path / f"{uuid}.mp4"), fourcc, 10, (rendered_images[0].shape[1], rendered_images[0].shape[0]))
    for image in rendered_images: video_writer.write(image)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("uuid", type=str, help="UUID of the video to visualize")
    parser.add_argument("--low", type=int, default=0, help="Start frame number")
    parser.add_argument("--high", type=int, default=1000, help="End frame number")
    parser.add_argument("--save-images", action="store_true", help="Save individual frames as images")
    args = parser.parse_args()

    visualize_video(args.uuid, low=args.low, high=args.high, save_images=args.save_images)