from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from pathlib import Path
import numpy as np
import cv2
from easysort.trainers.verdis_belt_motion import VerdisBeltGroundTruth

def file_to_saved_img_func(registry: RegistryBase, file: Path, expected_save_path: Path):
    cv2.imwrite(expected_save_path / file.name, registry.GET(file, registry.DefaultMarkers.ORIGINAL_MARKER))

if __name__ == "__main__":
    # Train category model
    CATEGORIES = ["plastics", "hard_plastics", "cardboard", "paper", "folie", "empty"]

    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    dataloader = DataLoader(registry, classes=CATEGORIES, destination=Path("verdis_category_dataset_reworked2"))
    dataloader.from_yolo_dataset(Path("verdis_category_dataset"))
    dataloader.from_registry(VerdisBeltGroundTruth, _label_json_to_category_func=lambda x: x.category, file_to_saved_img_func=file_to_saved_img_func, prefix="verdis/gadstrup/5")
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_category_reworked2", CATEGORIES, dataloader=dataloader)
    trainer.train(epochs=10, patience=5, imgsz=224, batch=32)

    trainer.eval()