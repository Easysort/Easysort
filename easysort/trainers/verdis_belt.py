from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from pathlib import Path
import numpy as np
import cv2
from easysort.trainers.verdis_belt_motion import VerdisBeltGroundTruth

POLY_POINTS = [(1418, 1512), (1019, 1502), (900, 704), (873, 76), (1000, 78), (1099, 747), (1406, 1510)]
BBOX = (min(p[0] for p in POLY_POINTS), min(p[1] for p in POLY_POINTS), max(p[0] for p in POLY_POINTS), max(p[1] for p in POLY_POINTS))

def crop_belt(img: np.ndarray, pad: int = 20) -> np.ndarray:
    """Crop image to belt bounding box."""
    x1, y1, x2, y2 = BBOX
    h, w = img.shape[:2]
    return img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

def file_to_saved_img_func(registry: RegistryBase, file: Path, expected_save_path: Path):
    img = np.array(registry.GET(file, registry.DefaultMarkers.ORIGINAL_MARKER))
    cropped_img = crop_belt(img)
    cv2.imwrite(expected_save_path / file.name, cropped_img)

if __name__ == "__main__":
    # Train category model
    CATEGORIES = ["plastics", "hard_plastics", "cardboard", "paper", "folie", "empty"]

    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    dataloader = DataLoader(registry, classes=CATEGORIES, destination=Path("verdis_category_dataset_reworked2"))
    dataloader.from_yolo_dataset(Path("verdis_category_dataset"))
    dataloader.from_registry(VerdisBeltGroundTruth, _label_json_to_category_func=lambda x: str(x.category).lower().replace(" ", "_"), file_to_saved_img_func=file_to_saved_img_func, prefix="verdis/gadstrup/5")
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_category_reworked2", CATEGORIES, dataloader=dataloader)
    trainer.train(epochs=10, patience=5, imgsz=224, batch=32)

    trainer.eval()