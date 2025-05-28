
from easysort.utils.image_sample import ImageSample
import os
import json
from dataclasses import asdict
from tqdm import tqdm

IMAGE_REGISTRY_PATH = "/Users/lucasvilsen/Documents/fun/easysort/image_registry"

def unpack_images():
    for image_path in tqdm(os.listdir(IMAGE_REGISTRY_PATH)):
        full_image_path = os.path.join(IMAGE_REGISTRY_PATH, image_path)
        os.makedirs(f"image_registry_v2/{image_path}", exist_ok=True)
        for sample_path in os.listdir(full_image_path):
            j = sample_path.split(".")[0]
            json_str = open(os.path.join(full_image_path, sample_path)).read()
            sample = ImageSample.from_json(json_str)
            if sample is None: continue
            image = sample.image.convert('L')
            detections = sample.detections
            metadata = sample.metadata

            image.save(f"image_registry_v2/{image_path}/{j}.png")
            predictions = {"detections": [det.to_json() for det in detections], "metadata": asdict(metadata)}
            with open(f"image_registry_v2/{image_path}/{j}.json", "w") as f:
                json.dump(predictions, f)

        

if __name__ == "__main__":
    unpack_images()
