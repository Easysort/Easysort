from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

from easysort.helpers import Concat
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.registry import RegistryBase
from easysort.sampler import Crop


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "easyprod" / "products" / "recycling" / "configs"


def latest_mp4(videos: list[Path]) -> Path | None:
  if not videos:
    return None
  return max(videos, key=lambda p: Concat._ts(p) or datetime.min)


def device_from_key(p: Path) -> str:
  if len(p.parts) >= 6:
    return p.parts[-6]
  if len(p.parts) >= 2:
    return p.parts[1]
  return p.parts[0] if p.parts else "unknown"


def registry_prefix_from_key(p: Path) -> str:
  return p.parts[0] if p.parts else ""


def crop_to_dict(crop: Crop) -> dict[str, int]:
  return {"x": crop.x, "y": crop.y, "w": crop.w, "h": crop.h}


def load_recycling_crop_configs(config_dir: Path = CONFIGS_DIR) -> list[tuple[Path, dict]]:
  configs: list[tuple[Path, dict]] = []
  for path in sorted(config_dir.glob("*.json")):
    try:
      config_json = json.loads(path.read_text())
    except json.JSONDecodeError:
      continue
    if isinstance(config_json, dict) and isinstance(config_json.get("crops"), dict):
      configs.append((path, config_json))
  return configs


def _config_priority(path: Path, config_json: dict, registry_prefix: str) -> tuple[int, int, str]:
  if registry_prefix and path.stem == registry_prefix:
    prefix_rank = 0
  elif registry_prefix and config_json.get("registry_prefix") == registry_prefix:
    prefix_rank = 1
  else:
    prefix_rank = 2
  validation_rank = 1 if "cameras" in config_json else 0
  return prefix_rank, validation_rank, path.name


def find_recycling_crop_config(
  device_name: str,
  registry_prefix: str,
  config_dir: Path = CONFIGS_DIR,
) -> tuple[Path | None, Crop | None]:
  configs = load_recycling_crop_configs(config_dir)
  exact_matches: list[tuple[tuple[int, int, str], Path, dict]] = []
  fallback_matches: list[tuple[tuple[int, int, str], Path, dict]] = []
  for path, config_json in configs:
    priority = _config_priority(path, config_json, registry_prefix)
    if device_name in config_json["crops"]:
      exact_matches.append((priority, path, config_json))
    elif priority[0] < 2:
      fallback_matches.append((priority, path, config_json))
  matches = sorted(exact_matches or fallback_matches, key=lambda item: item[0])
  if not matches:
    return None, None
  _, path, config_json = matches[0]
  crop_json = config_json["crops"].get(device_name)
  return path, (Crop(**crop_json) if crop_json else None)


def overwrite_crop_in_config(config_path: Path, device_name: str, crop: Crop) -> None:
  config_json = json.loads(config_path.read_text())
  config_json.setdefault("crops", {})[device_name] = crop_to_dict(crop)
  config_path.write_text(json.dumps(config_json, indent=2) + "\n")


def should_overwrite_crop(device_name: str, config_path: Path, crop: Crop) -> bool:
  prompt = f"Overwrite crop for {device_name} in {config_path.name} with {json.dumps(crop_to_dict(crop))}? [y/N]: "
  try:
    answer = input(prompt).strip().lower()
  except EOFError:
    return False
  return answer in {"y", "yes", "ok"}


def list_videos(registry: RegistryBase, prefix: str, device_filter: str | None, latest_only: bool) -> list[Path]:
  files = registry.LIST(prefix, suffix=[".mp4"])
  by_device: dict[str, list[Path]] = {}
  for f in files:
    dev = device_from_key(f)
    if device_filter and device_filter.lower() not in dev.lower():
      continue
    by_device.setdefault(dev, []).append(f)
  if latest_only:
    return [v for vs in by_device.values() if (v := latest_mp4(vs))]
  return [v for vs in by_device.values() for v in sorted(vs)]


def mmss(s: float) -> str:
  s = max(0, int(s))
  m, s = divmod(s, 60)
  h, m = divmod(m, 60)
  return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("prefix", nargs="?", default="argo")
  parser.add_argument("--device", default=None)
  parser.add_argument("--all", action="store_true", help="Include all matching videos (default: latest per device)")
  args = parser.parse_args()

  Registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  print("Keys: n=next video, a=next device, q=quit | click 2x to set crop, then confirm in terminal to save")
  videos = list_videos(Registry, args.prefix, args.device, latest_only=not args.all)
  if not videos:
    raise SystemExit("No videos found")

  by_device: dict[str, list[Path]] = {}
  for v in sorted(videos, key=lambda p: str(p)):
    by_device.setdefault(device_from_key(v), []).append(v)

  for device_name, device_videos in by_device.items():
    skip_device = False
    for video_key in device_videos:
      config_path, crop = find_recycling_crop_config(device_name, registry_prefix_from_key(video_key))
      original_crop = crop
      with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / video_key.name
        Registry.backend.GET_FILE(video_key, local)
        cap = cv2.VideoCapture(str(local))
        if not cap.isOpened():
          print("Failed:", video_key)
          continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / 2)))  # ~2 fps
        total_s = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        win = f"{device_name} | {video_key}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        state = {"p1": None, "crop": crop}
        action = None

        def click(ev, x, y, *_):
          if ev != cv2.EVENT_LBUTTONDOWN:
            return
          if state["p1"] is None:
            state["p1"] = (x, y)
            return
          x1, y1, x2, y2 = min(state["p1"][0], x), min(state["p1"][1], y), max(state["p1"][0], x), max(state["p1"][1], y)
          if x2 > x1 and y2 > y1:
            state["crop"] = Crop(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
            c = state["crop"]
            print(f'{{"x": {c.x}, "y": {c.y}, "w": {c.w}, "h": {c.h}}}')
          state["p1"] = None

        cv2.setMouseCallback(win, click)
        advance = False
        while not advance:
          cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
          while True:
            ok, frame = cap.read()
            if not ok:
              break
            H, W = frame.shape[:2]
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
            crop = state["crop"]
            if crop:
              x1, y1 = max(0, crop.x), max(0, crop.y)
              x2, y2 = min(W, crop.x + crop.w), min(H, crop.y + crop.h)
              if x1 < x2 and y1 < y2:
                roi = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                s = min((W // 3) / roi.shape[1], (H // 3) / roi.shape[0])
                inset = cv2.resize(roi, (max(1, int(roi.shape[1] * s)), max(1, int(roi.shape[0] * s))))
                frame[: inset.shape[0], : inset.shape[1]] = inset
                cv2.rectangle(frame, (0, 0), (inset.shape[1], inset.shape[0]), (0, 255, 0), 2)
            cv2.putText(frame, f"{mmss(t)} / {mmss(total_s)}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            k = cv2.waitKey(500) & 0xFF
            if k == ord("q"):
              raise SystemExit
            if k == ord("n"):
              action = "next_video"
              advance = True
              break
            if k == ord("a"):
              action = "next_device"
              skip_device = True
              advance = True
              break
            for _ in range(step - 1):
              if not cap.grab():
                ok = False
                break
            if not ok:
              break
        cap.release()
        cv2.destroyWindow(win)
      updated_crop = state["crop"]
      if config_path and updated_crop and updated_crop != original_crop:
        if should_overwrite_crop(device_name, config_path, updated_crop):
          overwrite_crop_in_config(config_path, device_name, updated_crop)
          print(f"Saved crop for {device_name} to {config_path.name}")
        else:
          print(f"Skipped saving crop for {device_name}")
      elif action and updated_crop and not config_path:
        print(f"No recycling config found for {device_name}; crop not saved")
      if skip_device:
        break
  cv2.destroyAllWindows()
