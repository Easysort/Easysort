import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from easysort.common.image_registry import ImageRegistry
from easysort.sorting.pipeline import SortingPipeline
from easysort.visualize.helpers import visualize_sorting_pipeline_image


def visualize_video(uuid: str, save_images: bool = False, rerun_pipeline: bool = True, plot_detections: bool = True):
    pipeline = SortingPipeline()
    # supabase_helper = SupabaseHelper(Environment.SUPABASE_AI_IMAGES_BUCKET)
    image_registry = ImageRegistry()
    if not image_registry.exists(uuid):
        print(f"Video {uuid} not found in Image Registry")
        return
    # if supabase_helper.exists(uuid):
    #     video_sample = supabase_helper.get(uuid)
    if image_registry.exists(uuid):
        video_sample = image_registry.compress_image_samples_to_video(uuid, delete=False)
    else:
        print(f"Video {uuid} not found in Supabase or Image Registry")
        return

    rendered_images = []
    for image_sample in tqdm(video_sample.samples.values(), desc="Visualizing video"):
        image = np.array(image_sample.image)
        detections = pipeline(image) if rerun_pipeline else image_sample.detections
        detections = detections if plot_detections else []
        main_view = visualize_sorting_pipeline_image(image, detections, show_plot=False)
        rendered_images.append(main_view)

    rendered_images_path = Path(f"rendered_videos/{uuid}")
    rendered_images_path.mkdir(parents=True, exist_ok=True)
    if save_images:
        for i, image in enumerate(rendered_images):
            cv2.imwrite(str(rendered_images_path / f"{i}.jpg"), image)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(rendered_images_path / f"{uuid}.mp4"), fourcc, 24, (rendered_images[0].shape[1], rendered_images[0].shape[0]))
    for image in rendered_images:
        video_writer.write(image)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("uuid", type=str, help="UUID of the video to visualize")
    parser.add_argument("--save-images", action="store_true", help="Save individual frames as images")
    parser.add_argument("--rerun-pipeline", action="store_true", help="Rerun the pipeline")
    parser.add_argument("--dont-plot-detections", action="store_true", help="Don't plot detections")
    args = parser.parse_args()

    visualize_video(args.uuid, save_images=args.save_images, rerun_pipeline=args.rerun_pipeline, plot_detections=not args.dont_plot_detections)
