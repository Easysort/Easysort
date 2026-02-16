from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import random
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore

from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP, current_timestamp
from easysort.validators.verdis_bales import VerdisBalesGroundTruth


def crop_bales(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
  h, w = img.shape[:2]
  crop_x = int(w * 0.2)
  crop_w = int(w * 0.4)
  crop_y = int(h * 0.6)
  crop_h = int(h * 0.4)
  if crop_w <= 0 or crop_h <= 0:
    return img, (0, 0, w, h)
  cropped = img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
  return cropped, (crop_x, crop_y, crop_w, crop_h)


def to_points(points: List[List[float]]) -> List[List[float]]:
  if not points:
    return []
  if len(points[0]) == 2:
    return points
  return [[(p[0] + p[2]) / 2, (p[1] + p[3]) / 2] for p in points]


def write_pose_label(label_path: Path, points: List[List[float]], crop_rect: Tuple[int, int, int, int]):
  crop_x, crop_y, crop_w, crop_h = crop_rect
  if crop_w <= 0 or crop_h <= 0:
    label_path.write_text("")
    return
  box_w = crop_w * 0.08
  box_h = crop_h * 0.08
  lines = []
  for x, y in points:
    if x < crop_x or y < crop_y or x > crop_x + crop_w or y > crop_y + crop_h:
      continue
    cx = (x - crop_x) / crop_w
    cy = (y - crop_y) / crop_h
    w = box_w / crop_w
    h = box_h / crop_h
    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {cx:.6f} {cy:.6f} 2")
  label_path.write_text("\n".join(lines))


def build_dataset(registry: RegistryBase, destination: Path, prefix: str = "verdis/gadstrup/4") -> Path:
  images_dir = destination / "images"
  labels_dir = destination / "labels"
  for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

  files = registry.LIST(prefix=prefix, check_exists_with_type=VerdisBalesGroundTruth)
  print(f"Found {len(files)} annotated files")

  for file in tqdm(files, desc="Building dataset"):
    gt = registry.GET(file, VerdisBalesGroundTruth, throw_error=False)
    if gt is None:
      continue
    points = to_points(gt.points)
    img = np.array(registry.GET(file, registry.DefaultMarkers.ORIGINAL_MARKER))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cropped, crop_rect = crop_bales(img)

    split = "train" if random.random() < 0.8 else "val"
    image_path = images_dir / split / file.name
    label_path = labels_dir / split / (file.stem + ".txt")

    cv2.imwrite(str(image_path), cropped)
    write_pose_label(label_path, points, crop_rect)

  data_yaml = destination / "data.yaml"
  data_yaml.write_text(
    "\n".join([
      f"path: {destination}",
      "train: images/train",
      "val: images/val",
      "names:",
      "  0: bale",
      "nc: 1",
      "kpt_shape: [1, 3]",
      "flip_idx: [0]",
    ])
  )
  return data_yaml


if __name__ == "__main__":
  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  dataset_dir = Path("verdis_bales_pose_dataset")
  data_yaml = build_dataset(registry, dataset_dir)

  model = YOLO("yolo11s-pose.pt")
  model.train(
    data=str(data_yaml),
    epochs=20,
    patience=10,
    imgsz=640,
    batch=16,
    project="verdis_bales_model",
    name="train",
    exist_ok=True,
    verbose=True,
    plots=True,
  )

  model.val(
    data=str(data_yaml),
    imgsz=640,
    batch=16,
    project="verdis_bales_model",
    name="val",
    verbose=True,
    plots=True,
  )
