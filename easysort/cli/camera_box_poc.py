import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import click

from easysort.sorting.pipeline import SortingPipeline
from easysort.utils.detections import Detection


@dataclass
class Metadata:
    timestamp_captured: float
    timestamp_processed: float
    dataset_uuid: str
    image_filename: str
    disk_free_gb: float
    detections: list[Detection]
    operator_message: str


@click.command()
@click.option("--interval", default=0.0, help="Interval between detections in seconds")
@click.option("--images-per-video", default=1, type=int, help="Number of images per video")
@click.option("--video-dir", default=None, type=click.Path(exists=True), help="Directory in which to save videos")
@click.option("--upload", is_flag=True, help="Upload videos to the cloud")
def run(interval: float, images_per_video: int, video_dir: Path | None, upload: bool) -> None:
    """
    Run the data collector.
    """
    pipeline = SortingPipeline(use_yolo_world=True)
    last_run = time.time()
    while True:
        for detections, _ in islice(pipeline.stream(use_depth=False), images_per_video):
            print("----------DETECTIONS------------")
            for detection in detections:
                print(detection)
            print("--------------------------------")

            elapsed_time = time.time() - last_run
            if elapsed_time < interval:
                time.sleep(interval - elapsed_time)
            elif interval > 0:
                print(f"Warning: Cannot keep up with interval of {interval} seconds (took {elapsed_time:.3f} seconds)")
            last_run = time.time()
        video = pipeline.image_registry.convert_to_video(pipeline.image_registry.uuid)
        if video_dir:
            video_path = Path(video_dir) / video.metadata.uuid
            video.save(video_path)
            print(f"Saved video to {video_path}")
        if upload:
            pipeline.image_registry.supabase_helper.upload_sample(video)
            print(f"Uploaded video {video.metadata.uuid} to cloud storage")
