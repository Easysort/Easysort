from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from easysort.trainer import ModelSchema, Trainer, MODELS_DIR
from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP, current_timestamp, unpack_video
from easysort.runner import Runner, extract_person_crops_from_video, PersonCrop
from easyprod.products.recycling.people_validation import (
  RECYCLING_PEOPLE_GT_ID,
  RECYCLING_PEOPLE_PSEUDO_GT_ID,
  RecyclingPeopleGT,
  RecyclingPeoplePseudoGT,
  camera_from_path,
  list_recycling_videos_by_camera,
  load_people_validation_config,
)

SCHEMA = ModelSchema(
  name="recycling_people",
  task="detect",
  classes=["person"],
  base_model="yolo11n.pt",
  weights_path=MODELS_DIR / "recycling_people.pt",
  imgsz=640,
)

@dataclass
class PersonCheck:
  has_person: bool


def _save_dataset_image(frame_bgr: np.ndarray, destination: Path):
  rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  Image.fromarray(rgb).save(destination, format="JPEG", quality=95)


def list_videos_balanced(registry: RegistryBase, n_per_camera: int = 50, config_name: str | None = None) -> Dict[str, List[Path]]:
  config = load_people_validation_config(config_name)
  by_camera = list_recycling_videos_by_camera(registry, config)
  counts = {cam: len(vs) for cam, vs in by_camera.items()}
  print(f"Videos per camera: {counts}")
  min_avail = min((len(vs) for vs in by_camera.values() if vs), default=0)
  if min_avail == 0:
    print("No videos found for any camera")
    return {}
  n = min(n_per_camera, min_avail)
  sampled_cameras = sum(1 for vs in by_camera.values() if vs)
  print(f"Sampling {n} videos per camera ({n * sampled_cameras} total)")
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


def label_videos(registry: RegistryBase, n_per_camera: int = 50, config_name: str | None = None) -> int:
  config = load_people_validation_config(config_name)
  min_w, min_h = config.min_w, config.min_h
  gpt_runner = Runner()
  videos_by_camera = list_videos_balanced(registry, n_per_camera, config_name=config_name)
  labeled, skip_show = 0, False

  for camera, videos in videos_by_camera.items():
    print(f"\n--- {camera}: {len(videos)} videos ---")

    for video_path in tqdm(videos, desc=f"Labeling {camera}"):
      if registry.EXISTS(video_path, RecyclingPeopleGT):
        labeled += 1
        continue

      person_crops = extract_person_crops_from_video(video_path, None, min_w=min_w, min_h=min_h, pad=0.08)

      if not person_crops:
        registry.POST(
          video_path,
          RecyclingPeopleGT(
            frame_bboxes={},
            metadata=RegistryBase.BaseDefaultTypes.BASEMETADATA(model="yolo11n-pose+gpt", created_at=current_timestamp()),
          ),
          RecyclingPeopleGT,
        )
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

      registry.POST(
        video_path,
        RecyclingPeopleGT(
          frame_bboxes=frame_bboxes,
          metadata=RegistryBase.BaseDefaultTypes.BASEMETADATA(model="yolo11n-pose+gpt", created_at=current_timestamp()),
        ),
        RecyclingPeopleGT,
      )
      labeled += 1

  print(f"\nLabeled {labeled} videos total")
  return labeled


def build_dataset(
  registry: RegistryBase,
  destination: Path,
  config_name: str | None = None,
  label_type=RecyclingPeopleGT,
) -> Path:
  config = load_people_validation_config(config_name)
  cameras = list(config.cameras)
  images_dir, labels_dir = destination / "images", destination / "labels"
  for split in ["train", "val"]:
    (images_dir / split).mkdir(parents=True, exist_ok=True)
    (labels_dir / split).mkdir(parents=True, exist_ok=True)

  all_videos = registry.LIST(config.registry_prefix, suffix=[".mp4"], check_exists_with_type=label_type)
  gt_videos = [Path(v) for v in all_videos if camera_from_path(v) in cameras]
  print(f"Found {len(gt_videos)} labeled videos")

  for video_path in tqdm(gt_videos, desc="Building dataset"):
    gt = registry.GET(video_path, label_type, throw_error=False)
    if gt is None or not gt.frame_bboxes:
      continue

    frames = unpack_video(cv2.VideoCapture(registry.backend.URL(video_path)))

    for frame_idx, bboxes in gt.frame_bboxes.items():
      if frame_idx >= len(frames):
        continue
      frame = frames[frame_idx]
      h, w = frame.shape[:2]

      split = "train" if random.random() < 0.8 else "val"
      name = f"{camera_from_path(video_path)}_{video_path.stem}_{frame_idx}"
      _save_dataset_image(frame, images_dir / split / f"{name}.jpg")

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


def build_pseudo_dataset(registry: RegistryBase, destination: Path, config_name: str | None = None) -> Path:
  registry.add_id(RecyclingPeoplePseudoGT, RECYCLING_PEOPLE_PSEUDO_GT_ID)
  return build_dataset(
    registry,
    destination,
    config_name=config_name,
    label_type=RecyclingPeoplePseudoGT,
  )


if __name__ == "__main__":
  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  registry.add_id(RecyclingPeopleGT, RECYCLING_PEOPLE_GT_ID)
  registry.add_id(RecyclingPeoplePseudoGT, RECYCLING_PEOPLE_PSEUDO_GT_ID)

  label_videos(registry, n_per_camera=50)
#   data_yaml = build_dataset(registry, Path("recycling_people_dataset"))

#   trainer = Trainer(SCHEMA, dataset=data_yaml)
#   trainer.train(epochs=20, patience=10, batch=16)
#   trainer.eval()
