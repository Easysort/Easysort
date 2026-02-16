"""Estimate bale production by motion between two frames."""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP


PREFIX = "verdis/gadstrup/4"
BALE_LENGTH_PX = 140
SAMPLE_TRIES = 200
MOTION_THRESHOLD_MULT = 0.001
FLOW_MAG_THRESHOLD = 2.5
FLOW_STEP = 10
ANGLE_TOLERANCE_DEG = 35.0


@dataclass
class MotionSample:
  folder_a: Path
  folder_b: Path
  img_a: Path
  img_b: Path


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


def list_folders_with_images(registry: RegistryBase, prefix: str) -> List[Path]:
  files = registry.backend.LIST(prefix)
  folders = {}
  for f in files:
    if f.suffix.lower() in (".jpg", ".png", ".jpeg"):
      folders.setdefault(f.parent, []).append(f)
  ordered = []
  for folder, imgs in folders.items():
    ordered.append((folder, sorted(imgs)))
  ordered.sort(key=lambda x: str(x[0]))
  return [f for f, _ in ordered]


def get_first_image(registry: RegistryBase, folder: Path) -> Optional[Path]:
  files = registry.backend.LIST(str(folder))
  imgs = sorted([f for f in files if f.suffix.lower() in (".jpg", ".png", ".jpeg")])
  return imgs[0] if imgs else None


def sample_adjacent_folders(registry: RegistryBase) -> Optional[MotionSample]:
  folders = list_folders_with_images(registry, PREFIX)
  if len(folders) < 2:
    return None
  for _ in range(SAMPLE_TRIES):
    idx = random.randint(0, len(folders) - 2)
    folder_a = folders[idx]
    folder_b = folders[idx + 1]
    img_a = get_first_image(registry, folder_a)
    img_b = get_first_image(registry, folder_b)
    if img_a and img_b:
      return MotionSample(folder_a, folder_b, img_a, img_b)
  return None


def build_adjacent_pairs(registry: RegistryBase) -> List[MotionSample]:
  folders = list_folders_with_images(registry, PREFIX)
  pairs = []
  for i in range(len(folders) - 1):
    folder_a = folders[i]
    folder_b = folders[i + 1]
    img_a = get_first_image(registry, folder_a)
    img_b = get_first_image(registry, folder_b)
    if img_a and img_b:
      pairs.append(MotionSample(folder_a, folder_b, img_a, img_b))
  return pairs


def to_bgr(registry: RegistryBase, key: Path) -> np.ndarray:
  img = np.array(registry.GET(key, registry.DefaultMarkers.ORIGINAL_MARKER))
  return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def build_motion_mask(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
  diff = cv2.absdiff(gray1, gray2)
  diff = cv2.GaussianBlur(diff, (5, 5), 0)
  _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
  mask = cv2.dilate(mask, kernel, iterations=1)
  return mask


def estimate_motion_flow(
  img1: np.ndarray,
  img2: np.ndarray,
  mag_thresh: float = FLOW_MAG_THRESHOLD,
) -> Tuple[float, float, List[Tuple[int, int, int, int]], List[float]]:
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  mask = build_motion_mask(gray1, gray2)
  flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 4, 35, 5, 7, 1.5, 0)  # type: ignore[arg-type]
  h, w = gray1.shape[:2]
  arrows: List[Tuple[int, int, int, int]] = []
  mags: List[float] = []
  angles: List[float] = []
  for y in range(0, h, FLOW_STEP):
    for x in range(0, w, FLOW_STEP):
      if mask[y, x] == 0:
        continue
      fx, fy = flow[y, x]
      mag = float(np.hypot(fx, fy))
      if mag < mag_thresh:
        continue
      if fx <= 0 or fy <= 0:
        continue
      x2 = int(x + fx)
      y2 = int(y + fy)
      arrows.append((x, y, x2, y2))
      mags.append(mag)
      angles.append(float(np.arctan2(fy, fx)))

  if not arrows:
    return 0.0, 0.0, [], []

  med_angle = float(np.median(angles)) if angles else 0.0
  tol = np.deg2rad(ANGLE_TOLERANCE_DEG)
  filtered = []
  filtered_mags = []
  for (x1, y1, x2, y2), ang, mag in zip(arrows, angles, mags):
    if abs(ang - med_angle) <= tol:
      filtered.append((x1, y1, x2, y2))
      filtered_mags.append(mag)

  if not filtered:
    filtered = arrows
    filtered_mags = mags

  dxs = [x2 - x1 for x1, y1, x2, y2 in filtered]
  dys = [y2 - y1 for x1, y1, x2, y2 in filtered]
  dx = float(np.median(dxs))
  dy = float(np.median(dys))
  return dx, dy, filtered, filtered_mags


def bale_count_from_motion(dx: float, dy: float, bale_length_px: float) -> int:
  if dx <= 0 or dy <= 0:
    return 0
  dist = float(np.hypot(dx, dy))
  if dist <= 0:
    return 0
  return int(dist / bale_length_px)


def render_histogram(values: List[float], width: int, height: int, bins: int = 20) -> np.ndarray:
  img = np.zeros((height, width, 3), dtype=np.uint8)
  if not values:
    cv2.putText(img, "No motion", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return img
  v = np.array(values, dtype=np.float32)
  max_v = float(np.max(v)) if v.size else 1.0
  hist, _ = np.histogram(v, bins=bins, range=(0, max_v if max_v > 0 else 1.0))
  hist = hist.astype(np.float32)
  hist = hist / hist.max() if hist.max() > 0 else hist
  bar_w = max(1, width // bins)
  for i, h in enumerate(hist):
    x1 = i * bar_w
    x2 = min(width - 1, x1 + bar_w - 1)
    bar_h = int(h * (height - 20))
    cv2.rectangle(img, (x1, height - 1), (x2, height - 1 - bar_h), (0, 200, 255), -1)
  cv2.putText(img, "Motion dist", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
  return img


def render_sample(registry: RegistryBase, sample: MotionSample, rank: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
  if sample is None:
    print("No adjacent folders found")
    return None

  img1 = to_bgr(registry, sample.img_a)
  img2 = to_bgr(registry, sample.img_b)

  crop1, _ = crop_bales(img1)
  crop2, _ = crop_bales(img2)

  dx, dy, arrows_vecs, motion_vals = estimate_motion_flow(crop1, crop2, mag_thresh=FLOW_MAG_THRESHOLD)
  bale_count = bale_count_from_motion(dx, dy, BALE_LENGTH_PX)

  print(f"Folder A: {sample.folder_a}")
  print(f"Folder B: {sample.folder_b}")
  print(f"Estimated translation: dx={dx:.2f}, dy={dy:.2f}")
  print(f"Estimated bales: {bale_count}")

  h = min(crop1.shape[0], crop2.shape[0])
  w = min(crop1.shape[1], crop2.shape[1])
  crop1 = crop1[:h, :w]
  crop2 = crop2[:h, :w]

  overlay = cv2.addWeighted(crop1, 0.5, crop2, 0.5, 0)
  arrows = overlay.copy()
  for x1, y1, x2, y2 in arrows_vecs:
    cv2.arrowedLine(arrows, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.2)

  info = f"dx={dx:.1f}, dy={dy:.1f}, bales={bale_count}"
  if rank:
    info = f"{rank[0]}/{rank[1]}  " + info
  cv2.putText(arrows, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

  grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
  grid[0:h, 0:w] = crop1
  grid[0:h, w : 2 * w] = crop2
  grid[h : 2 * h, 0:w] = arrows
  grid[h : 2 * h, w : 2 * w] = render_histogram(motion_vals, w, h)

  cv2.putText(grid, "A", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(grid, "B", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(grid, "Arrows", (10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(grid, "Histogram", (w + 10, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

  return grid


def rank_samples_by_motion(registry: RegistryBase, min_dist: float, max_samples: Optional[int] = None) -> List[Tuple[MotionSample, float, float, float]]:
  pairs = build_adjacent_pairs(registry)
  if max_samples is not None and max_samples > 0 and len(pairs) > max_samples:
    pairs = random.sample(pairs, max_samples)
  ranked = []
  for sample in tqdm(pairs, desc="Fetching from registry + scoring motion"):
    img1 = to_bgr(registry, sample.img_a)
    img2 = to_bgr(registry, sample.img_b)
    crop1, _ = crop_bales(img1)
    crop2, _ = crop_bales(img2)
    dx, dy, _, _ = estimate_motion_flow(crop1, crop2, mag_thresh=FLOW_MAG_THRESHOLD)
    if dx <= 0 or dy <= 0:
      dist = 0.0
    else:
      dist = float(np.hypot(dx, dy))
    if dist >= min_dist:
      ranked.append((sample, dx, dy, dist))
  ranked.sort(key=lambda x: x[3], reverse=True)
  return ranked


def main():
  import sys

  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  window_name = "Bales motion debug"

  use_motion = "--motion" in sys.argv
  min_dist = BALE_LENGTH_PX * MOTION_THRESHOLD_MULT
  max_samples = None
  if "-n" in sys.argv:
    try:
      idx = sys.argv.index("-n")
      max_samples = int(sys.argv[idx + 1])
    except Exception:
      max_samples = None
  if "--motion-min" in sys.argv:
    try:
      idx = sys.argv.index("--motion-min")
      min_dist = float(sys.argv[idx + 1])
    except Exception:
      pass

  if use_motion:
    ranked = rank_samples_by_motion(registry, min_dist, max_samples=max_samples)
    if not ranked:
      print("No samples above motion threshold")
      return
    i = 0
    while True:
      sample, dx, dy, dist = ranked[i]
      grid = render_sample(registry, sample, rank=(i + 1, len(ranked)))
      if grid is None:
        break
      cv2.imshow(window_name, grid)
      key = cv2.waitKey(0)
      if key in (ord("q"), ord("Q")):
        break
      if key == 32:
        i += 1
        if i >= len(ranked):
          break
    cv2.destroyAllWindows()
    return

  sample = sample_adjacent_folders(registry)
  if sample is None:
    print("No adjacent folders found")
    return
  grid = render_sample(registry, sample)
  if grid is None:
    return
  cv2.imshow(window_name, grid)

  while True:
    key = cv2.waitKey(0)
    if key in (ord("q"), ord("Q")):
      break
    if key == 32:
      sample = sample_adjacent_folders(registry)
      if sample is None:
        break
      grid = render_sample(registry, sample)
      if grid is None:
        break
      cv2.imshow(window_name, grid)

  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
