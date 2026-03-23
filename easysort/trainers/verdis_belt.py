from easysort.trainer import ModelSchema, Trainer, MODELS_DIR
from easysort.trainers.verdis_belt_motion import crop_belt, VerdisBeltGroundTruth
from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from pathlib import Path
import numpy as np
import cv2


def _preprocess(img: np.ndarray) -> np.ndarray:
  if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  return crop_belt(img)


SCHEMA = ModelSchema(
  name="verdis_category",
  task="classify",
  classes=sorted(["plastics", "hard_plastics", "cardboard", "paper", "folie", "empty"]),
  base_model="yolo11s-cls.pt",
  weights_path=MODELS_DIR / "verdis_category.pt",
  imgsz=224,
  preprocess=_preprocess,
)


def _file_to_saved_img(registry: RegistryBase, file: Path, expected_save_path: Path):
  img = np.array(registry.GET(file, registry.DefaultMarkers.ORIGINAL_MARKER))
  cv2.imwrite(expected_save_path / file.name, crop_belt(img))


if __name__ == "__main__":
  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  dataloader = DataLoader(registry, classes=SCHEMA.classes, destination=Path("verdis_category_dataset_reworked2"))
  dataloader.from_yolo_dataset(Path("verdis_category_dataset"))
  dataloader.from_registry(VerdisBeltGroundTruth, _label_json_to_category_func=lambda x: str(x.category).lower().replace(" ", "_"), file_to_saved_img_func=_file_to_saved_img, prefix="verdis/gadstrup/5")

  trainer = Trainer(SCHEMA, dataset=dataloader.destination)
  trainer.train(epochs=10, patience=5, batch=32)
  trainer.eval()
