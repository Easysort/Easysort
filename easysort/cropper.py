from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

from easysort.helpers import Concat
from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.sampler import DEVICE_TO_CROP, Crop
from easyprod.scripts.argo.argo import CROPS as ARGO_CROPS


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
  print("Keys: n=next video, a=next device, q=quit | click 2x to set crop")
  videos = list_videos(Registry, args.prefix, args.device, latest_only=not args.all)
  if not videos:
    raise SystemExit("No videos found")

  by_device: dict[str, list[Path]] = {}
  for v in sorted(videos, key=lambda p: str(p)):
    by_device.setdefault(device_from_key(v), []).append(v)

  for device_name, device_videos in by_device.items():
    skip_device = False
    for video_key in device_videos:
      crop = ARGO_CROPS.get(device_name) or DEVICE_TO_CROP.get(device_name)
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
              advance = True
              break
            if k == ord("a"):
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
      if skip_device:
        break
  cv2.destroyAllWindows()
