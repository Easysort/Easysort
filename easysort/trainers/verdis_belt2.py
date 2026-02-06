from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from pathlib import Path
import numpy as np
import cv2


if __name__ == "__main__":
    # Train category model
    CATEGORIES = ["plastics", "hard_plastics", "cardboard", "paper", "folie", "empty"]

    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    dataloader = DataLoader(registry, classes=CATEGORIES, destination=Path("verdis_category_dataset_reworked"))
    dataloader.from_yolo_dataset(Path("verdis_category_dataset"))
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_category_reworked", CATEGORIES, dataloader=dataloader)
    trainer.train(epochs=10, patience=5, imgsz=224, batch=32)

    trainer.eval()