from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from pathlib import Path
from typing import List
import cv2
import numpy as np
from uuid import uuid4
from dataclasses import dataclass, field
from easysort.helpers import current_timestamp


@dataclass
class VerdisBeltGroundTruth:
    category: str
    motion: bool
    id: str = field(default_factory=lambda: "5adf5d6f-539a-4533-9461-ff8b390fd9cf")
    metadata: RegistryBase.BaseDefaultTypes.BASEMETADATA = field(default_factory=lambda: RegistryBase.BaseDefaultTypes.BASEMETADATA(model="human", created_at=current_timestamp()))

# Single source of truth for belt geometry (shared by category + motion train and inference)
POLY_POINTS = [(1418, 1512), (1019, 1502), (900, 704), (873, 76), (1000, 78), (1099, 747), (1406, 1510)]
BBOX = (min(p[0] for p in POLY_POINTS), min(p[1] for p in POLY_POINTS), max(p[0] for p in POLY_POINTS), max(p[1] for p in POLY_POINTS))


def crop_belt(img: np.ndarray, pad: int = 20) -> np.ndarray:
    """Crop image to belt bounding box. Use this for both training and inference."""
    x1, y1, x2, y2 = BBOX
    h, w = img.shape[:2]
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def frames_to_motion_input(imgs: List[np.ndarray]) -> np.ndarray:
    """Same preprocessing as motion training: cropped absdiff of first and third frame."""
    if len(imgs) < 2:
        raise ValueError("Need at least 2 frames for motion input")
    img1 = np.asarray(imgs[0])
    img3 = np.asarray(imgs[2] if len(imgs) >= 3 else imgs[-1])
    diff = cv2.absdiff(img1, img3)
    return crop_belt(diff)


def file_to_saved_img_func(registry: RegistryBase, file: Path, expected_save_path: Path):
    neighboring_files = sorted([f for f in registry.backend.LIST(file.parent) if f.suffix.lower() in (".jpg", ".png", ".jpeg")])
    assert len(neighboring_files) == 3, "Expected 3 neighboring files"
    img1 = registry.GET(neighboring_files[0], registry.DefaultMarkers.ORIGINAL_MARKER)
    img3 = registry.GET(neighboring_files[2], registry.DefaultMarkers.ORIGINAL_MARKER)
    motion_input = frames_to_motion_input([img1, img3])
    cv2.imwrite(expected_save_path / file.name, motion_input)

if __name__ == "__main__":
    # Train motion model
    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    MOTION_CATEGORIES = ["motion", "no_motion"]

    dataloader = DataLoader(registry, classes=MOTION_CATEGORIES, destination=Path("verdis_motion_dataset_reworked"), force_recreate=True)
    print("here")
    dataloader.from_registry(VerdisBeltGroundTruth, _label_json_to_category_func=lambda x: "motion" if x.motion else "no_motion", file_to_saved_img_func=file_to_saved_img_func, prefix="verdis/gadstrup/5")
    print("here")
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_motion_reworked", MOTION_CATEGORIES, dataloader=dataloader)
    trainer.train(epochs=10, patience=5, imgsz=224, batch=32)

    trainer.eval()