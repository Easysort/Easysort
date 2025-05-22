from dataclasses import dataclass
import time
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
@click.option("--image-save-interval", default=0.0, help="Minimum interval between saving images in seconds")
def run(interval: float, image_save_interval: float) -> None:
    """
    Run the data collector.
    """
    pipeline = SortingPipeline(use_yolo_world=True)
    last_run = time.time()
    last_image_save = 0.0
    for detection in pipeline.stream(use_depth=False):
        print("------PICKED UP DETECTION--------")
        print(detection)
        print("--------------------------------")

        elapsed_time = time.time() - last_run
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
        elif interval > 0:
            print(f"Warning: Cannot keep up with interval of {interval} seconds (took {elapsed_time:.3f} seconds)")
        last_run = time.time()
