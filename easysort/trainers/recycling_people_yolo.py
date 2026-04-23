"""Minimal YOLO-n trainer for the recycling_people dataset.

Reuses the dataset building logic from `recycling_people.py` and trains a
`yolo11n.pt` detector via ultralytics (handled by `easysort.trainer.Trainer`).
"""
from pathlib import Path
from dataclasses import replace
import sys

from easysort.trainer import ModelSchema, Trainer, MODELS_DIR
from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.trainers.recycling_people import build_dataset, build_pseudo_dataset
from easyprod.products.recycling.people_validation import (
  RECYCLING_PEOPLE_GT_ID,
  RECYCLING_PEOPLE_PSEUDO_GT_ID,
  RecyclingPeopleGT,
  RecyclingPeoplePseudoGT,
)

SCHEMA = ModelSchema(
  name="recycling_people_yolo",
  task="detect",
  classes=["person"],
  base_model="yolo11n.pt",
  weights_path=MODELS_DIR / "recycling_people_yolo.pt",
  imgsz=640,
)
PSEUDO_SCHEMA = replace(
  SCHEMA,
  name="recycling_people_yolo_pseudo",
  weights_path=MODELS_DIR / "recycling_people_yolo_pseudo.pt",
)


def train_yolo(
  registry: RegistryBase,
  destination: Path,
  *,
  config_name: str | None = None,
  epochs: int = 30,
  patience: int = 10,
  batch: int = 16,
  use_pseudo: bool = False,
  weights_output: Path | None = None,
) -> Path:
  destination = Path(destination)
  if use_pseudo:
    data_yaml = build_pseudo_dataset(registry, destination, config_name=config_name)
    schema = PSEUDO_SCHEMA
  else:
    data_yaml = build_dataset(registry, destination, config_name=config_name, label_type=RecyclingPeopleGT)
    schema = SCHEMA
  if weights_output is not None:
    schema = replace(schema, weights_path=weights_output)

  trainer = Trainer(schema, dataset=data_yaml)
  print(f"Training YOLO-n people model from dataset: {data_yaml}")
  trainer.train(epochs=epochs, patience=patience, batch=batch)
  trainer.eval()

  best = Path(f"{schema.name}_model") / "train" / "weights" / "best.pt"
  if not best.exists():
    raise FileNotFoundError(f"Expected trained weights at {best}")
  schema.weights_path.parent.mkdir(parents=True, exist_ok=True)
  import shutil
  shutil.copy2(best, schema.weights_path)
  print(f"Copied trained weights to {schema.weights_path}")
  return schema.weights_path


if __name__ == "__main__":
  config_name = None
  dataset_path = Path("recycling_people_yolo_dataset")
  epochs = 30
  patience = 10
  batch = 16
  use_pseudo = False
  weights_output = None

  for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
      config_name = sys.argv[i + 1]
    elif arg == "--dataset" and i + 1 < len(sys.argv):
      dataset_path = Path(sys.argv[i + 1])
    elif arg == "--epochs" and i + 1 < len(sys.argv):
      epochs = int(sys.argv[i + 1])
    elif arg == "--patience" and i + 1 < len(sys.argv):
      patience = int(sys.argv[i + 1])
    elif arg == "--batch" and i + 1 < len(sys.argv):
      batch = int(sys.argv[i + 1])
    elif arg == "--weights-output" and i + 1 < len(sys.argv):
      weights_output = Path(sys.argv[i + 1])
    elif arg == "--pseudo":
      use_pseudo = True

  if "--help" in sys.argv or "-h" in sys.argv:
    print(
      "Usage: python -m easysort.trainers.recycling_people_yolo "
      "[--pseudo] [--dataset PATH] [--config NAME] [--epochs N] [--patience N] "
      "[--batch N] [--weights-output PATH]"
    )
    sys.exit(0)

  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  registry.add_id(RecyclingPeopleGT, RECYCLING_PEOPLE_GT_ID)
  registry.add_id(RecyclingPeoplePseudoGT, RECYCLING_PEOPLE_PSEUDO_GT_ID)
  train_yolo(
    registry,
    dataset_path,
    config_name=config_name,
    epochs=epochs,
    patience=patience,
    batch=batch,
    use_pseudo=use_pseudo,
    weights_output=weights_output,
  )
