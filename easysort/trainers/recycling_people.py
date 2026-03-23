from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
import random
import json
import cv2
import numpy as np
from tqdm import tqdm

from easysort.trainer import ModelSchema, Trainer, MODELS_DIR
from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP, current_timestamp, unpack_video
from easysort.sampler import Crop
from easysort.runner import Runner, extract_person_crops_from_video, PersonCrop

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOYMENT_FOLDER = "argo"

SCHEMA = ModelSchema(
  name="recycling_people",
  task="detect",
  classes=["person"],
  base_model="yolo11n.pt",
  weights_path=MODELS_DIR / "recycling_people.pt",
  imgsz=640,
)

RECYCLING_PEOPLE_GT_ID = "c3a7f1e2-8b4d-4e6a-9f5c-2d1b0a3e8c7f"

@dataclass
class RecyclingPeopleGT:
  frame_bboxes: Dict[int, List[List[int]]]
  id: str = field(default_factory=lambda: RECYCLING_PEOPLE_GT_ID)
  metadata: RegistryBase.BaseDefaultTypes.BASEMETADATA = field(
    default_factory=lambda: RegistryBase.BaseDefaultTypes.BASEMETADATA(model="yolo11n-pose+gpt", created_at=current_timestamp())
  )


@dataclass
class PersonCheck:
  has_person: bool


def _load_config(config_name: str = "kk"):
  config_path = PROJECT_ROOT / "easyprod" / "products" / "recycling" / "configs" / f"{config_name}.json"
  config = json.loads(config_path.read_text())
  crops = {k: Crop(**v) for k, v in config["crops"].items()}
  return crops, config.get("min_w", 80), config.get("min_h", 200)


def _camera(path: Path) -> str:
  return path.parts[-6] if len(path.parts) >= 6 else ""


def list_videos_balanced(registry: RegistryBase, n_per_camera: int = 50) -> Dict[str, List[Path]]:
  crops, *_ = _load_config()
  cameras = list(crops.keys())
  all_videos = registry.LIST(DEPLOYMENT_FOLDER, suffix=[".mp4"])

  by_camera: Dict[str, List[Path]] = {cam: [] for cam in cameras}
  for v in all_videos:
    cam = _camera(v)
    if cam in by_camera:
      by_camera[cam].append(v)

  counts = {cam: len(vs) for cam, vs in by_camera.items()}
  print(f"Videos per camera: {counts}")
  min_avail = min((len(vs) for vs in by_camera.values() if vs), default=0)
  if min_avail == 0:
    print("No videos found for any camera")
    return {}
  n = min(n_per_camera, min_avail)
  print(f"Sampling {n} videos per camera ({n * len(cameras)} total)")
  return {cam: random.sample(vs, n) for cam, vs in by_camera.items() if vs}


def _show_results(video_name: str, crops: List[PersonCrop], confirmations: List[PersonCheck]):
  """Display person crops with GPT confirmation: green=confirmed, red=rejected."""
  if not crops:
    return
  tiles = []
  for pc, conf in zip(crops, confirmations):
    img = cv2.resize(pc.image.copy(), (128, 256))
    color = (0, 255, 0) if conf.has_person else (0, 0, 255)
    label = "PERSON" if conf.has_person else "REJECTED"
    cv2.rectangle(img, (0, 0), (127, 255), color, 3)
    cv2.putText(img, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    tiles.append(img)
  per_row = min(len(tiles), 8)
  rows = []
  for i in range(0, len(tiles), per_row):
    row = tiles[i:i + per_row]
    while len(row) < per_row:
      row.append(np.zeros((256, 128, 3), dtype=np.uint8))
    rows.append(np.hstack(row))
  grid = np.vstack(rows)
  cv2.imshow(f"Results: {video_name}", grid)
  key = cv2.waitKey(2000) & 0xFF
  cv2.destroyAllWindows()
  return key == ord("q")


def label_videos(registry: RegistryBase, n_per_camera: int = 50) -> int:
  crops, min_w, min_h = _load_config()
  gpt_runner = Runner()
  videos_by_camera = list_videos_balanced(registry, n_per_camera)
  labeled, skip_show = 0, False

  for camera, videos in videos_by_camera.items():
    crop = crops[camera]
    print(f"\n--- {camera}: {len(videos)} videos ---")

    for video_path in tqdm(videos, desc=f"Labeling {camera}"):
      if registry.EXISTS(video_path, RecyclingPeopleGT):
        labeled += 1
        continue

      person_crops = extract_person_crops_from_video(video_path, crop, min_w=min_w, min_h=min_h, pad=0.08)

      if not person_crops:
        registry.POST(video_path, RecyclingPeopleGT(frame_bboxes={}), RecyclingPeopleGT)
        print(f"  {video_path.name}: no people detected")
        labeled += 1
        continue

      images = [[pc.image] for pc in person_crops]
      confirmations = gpt_runner.gpt(images, PersonCheck, "Is there a person in this image? Return JSON with has_person boolean.")

      frame_bboxes: Dict[int, List[List[int]]] = {}
      for pc, conf in zip(person_crops, confirmations):
        if conf.has_person:
          frame_bboxes.setdefault(pc.frame_idx, []).append(list(pc.box))

      confirmed = sum(len(bbs) for bbs in frame_bboxes.values())
      print(f"  {video_path.name}: {confirmed}/{len(person_crops)} confirmed in {len(frame_bboxes)} frames")

      if not skip_show:
        skip_show = _show_results(video_path.name, person_crops, confirmations)

      registry.POST(video_path, RecyclingPeopleGT(frame_bboxes=frame_bboxes), RecyclingPeopleGT)
      labeled += 1

  print(f"\nLabeled {labeled} videos total")
  return labeled


def build_dataset(registry: RegistryBase, destination: Path) -> Path:
  crops, *_ = _load_config()
  cameras = list(crops.keys())
  images_dir, labels_dir = destination / "images", destination / "labels"
  for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

  all_videos = registry.LIST(DEPLOYMENT_FOLDER, suffix=[".mp4"], check_exists_with_type=RecyclingPeopleGT)
  gt_videos = [v for v in all_videos if _camera(v) in cameras]
  print(f"Found {len(gt_videos)} labeled videos")

  for video_path in tqdm(gt_videos, desc="Building dataset"):
    gt = registry.GET(video_path, RecyclingPeopleGT, throw_error=False)
    if gt is None or not gt.frame_bboxes:
      continue

    crop = crops.get(_camera(video_path))
    if crop is None:
      continue

    frames = unpack_video(cv2.VideoCapture(registry.backend.URL(video_path)))
    frames = [f[crop.y:crop.y + crop.h, crop.x:crop.x + crop.w] for f in frames]

    for frame_idx, bboxes in gt.frame_bboxes.items():
      if frame_idx >= len(frames):
        continue
      frame = frames[frame_idx]
      h, w = frame.shape[:2]

      split = "train" if random.random() < 0.8 else "val"
      name = f"{_camera(video_path)}_{video_path.stem}_{frame_idx}"
      cv2.imwrite(str(images_dir / split / f"{name}.jpg"), frame)

      lines = []
      for x1, y1, x2, y2 in bboxes:
        cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
      (labels_dir / split / f"{name}.txt").write_text("\n".join(lines))

  data_yaml = destination / "data.yaml"
  data_yaml.write_text("\n".join([
    f"path: {destination}", "train: images/train", "val: images/val",
    "names:", "  0: person", "nc: 1",
  ]))
  return data_yaml


if __name__ == "__main__":
  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  registry.add_id(RecyclingPeopleGT, RECYCLING_PEOPLE_GT_ID)

  label_videos(registry, n_per_camera=50)
#   data_yaml = build_dataset(registry, Path("recycling_people_dataset"))

#   trainer = Trainer(SCHEMA, dataset=data_yaml)
#   trainer.train(epochs=20, patience=10, batch=16)
#   trainer.eval()
