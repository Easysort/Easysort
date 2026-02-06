from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from pathlib import Path

if __name__ == "__main__":
    CATEGORIES = ["Plastics", "Hard plastics", "Cardboard", "Paper", "Folie", "Empty"]

    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    dataloader = DataLoader(registry, classes=CATEGORIES)
    dataloader.from_yolo_dataset(Path("verdis_category_dataset"))
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_category", CATEGORIES)
    trainer.train(Path("verdis_category_dataset"), epochs=30, patience=5, imgsz=224, batch=32)

    trainer.eval()
    