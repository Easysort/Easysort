from pathlib import Path
from typing import List, Tuple
import random
import cv2
import numpy as np
from tqdm import tqdm

from easysort.trainer import ModelSchema, Trainer, MODELS_DIR
from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.validators.verdis_bales import VerdisBalesGroundTruth


SCHEMA = ModelSchema(
  name="verdis_bales",
  task="detect",
  classes=["bale"],
  base_model="rfdetr-l.pt",
  weights_path=MODELS_DIR / "verdis_bales.pt",
  imgsz=640,
)


def crop_bales(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
  h, w = img.shape[:2]
  crop_x, crop_w = int(w * 0.2), int(w * 0.4)
  crop_y, crop_h = int(h * 0.6), int(h * 0.4)
  if crop_w <= 0 or crop_h <= 0:
    return img, (0, 0, w, h)
  return img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (crop_x, crop_y, crop_w, crop_h)


def _to_points(points: List[List[float]]) -> List[List[float]]:
  if not points or len(points[0]) == 2:
    return points
  return [[(p[0] + p[2]) / 2, (p[1] + p[3]) / 2] for p in points]


def _write_label(label_path: Path, points: List[List[float]], crop_rect: Tuple[int, int, int, int]):
  crop_x, crop_y, crop_w, crop_h = crop_rect
  if crop_w <= 0 or crop_h <= 0:
    label_path.write_text("")
    return
  bw, bh = crop_w * 0.08 / crop_w, crop_h * 0.08 / crop_h
  lines = []
  for x, y in points:
    if x < crop_x or y < crop_y or x > crop_x + crop_w or y > crop_y + crop_h:
      continue
    cx, cy = (x - crop_x) / crop_w, (y - crop_y) / crop_h
    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
  label_path.write_text("\n".join(lines))


def build_dataset(registry: RegistryBase, destination: Path, prefix: str = "verdis/gadstrup/4") -> Path:
  images_dir, labels_dir = destination / "images", destination / "labels"
  for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

  files = registry.LIST(prefix=prefix, check_exists_with_type=VerdisBalesGroundTruth)
  print(f"Found {len(files)} annotated files")

  for file in tqdm(files, desc="Building dataset"):
    gt = registry.GET(file, VerdisBalesGroundTruth, throw_error=False)
    if gt is None:
      continue
    points = _to_points(gt.points)
    img = cv2.cvtColor(np.array(registry.GET(file, registry.DefaultMarkers.ORIGINAL_MARKER)), cv2.COLOR_RGB2BGR)
    cropped, crop_rect = crop_bales(img)
    split = "train" if random.random() < 0.8 else "val"
    cv2.imwrite(str(images_dir / split / file.name), cropped)
    _write_label(labels_dir / split / (file.stem + ".txt"), points, crop_rect)

  data_yaml = destination / "data.yaml"
  data_yaml.write_text("\n".join([
    f"path: {destination}", "train: images/train", "val: images/val",
    "names:", "  0: bale", "nc: 1",
  ]))
  return data_yaml


if __name__ == "__main__":
  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  data_yaml = build_dataset(registry, Path("verdis_bales_dataset"))

  trainer = Trainer(SCHEMA, dataset=data_yaml)
  trainer.train(epochs=20, patience=10, batch=16)
  trainer.eval()
