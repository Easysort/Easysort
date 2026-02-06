from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from easysort.validators.verdis_belt import VerdisBeltGroundTruth
from pathlib import Path
import cv2
from uuid import uuid4


def file_to_saved_img_func(registry: RegistryBase, file: Path, expected_save_path: Path):
    neighboring_files = sorted([f for f in registry.backend.LIST(file.parent) if f.suffix.lower() in (".jpg", ".png", ".jpeg")])
    assert len(neighboring_files) == 3, "Expected 3 neighboring files"
    img1 = registry.GET(neighboring_files[0], registry.DefaultMarkers.ORIGINAL_MARKER)
    img3 = registry.GET(neighboring_files[2], registry.DefaultMarkers.ORIGINAL_MARKER)
    diff_img = img3 - img1
    cv2.imwrite(expected_save_path / file.name, diff_img)
    

if __name__ == "__main__":
    # Train motion model
    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    MOTION_CATEGORIES = ["motion", "no_motion"]

    dataloader = DataLoader(registry, classes=MOTION_CATEGORIES, destination=Path("verdis_motion_dataset_reworked"), force_recreate=True)
    dataloader.from_registry(VerdisBeltGroundTruth, _label_json_to_category_func=lambda x: x.motion, file_to_saved_img_func=file_to_saved_img_func)
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_motion_reworked", MOTION_CATEGORIES, dataloader=dataloader)
    trainer.train(epochs=10, patience=5, imgsz=224, batch=32)

    trainer.eval()