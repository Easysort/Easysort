from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2

from easysort.helpers import Concat
from easysort.registry import Registry
from easysort.sampler import DEVICE_TO_CROP, Crop


def latest_mp4(device_dir: Path) -> Path | None:
    vids = [p for p in device_dir.rglob("*.mp4") if not p.name.startswith("._")]
    if not vids: return None
    return max(vids, key=lambda p: Concat._ts(p) or datetime.fromtimestamp(p.stat().st_mtime))

def mmss(s: float) -> str:
    s = max(0, int(s)); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

if __name__ == "__main__":
    root = Registry._registry_path(sys.argv[1] if len(sys.argv) > 1 else "argo")
    if not root.is_dir(): raise SystemExit(f"Not a folder: {root}")
    print("Keys: n=next, q=quit | click 2x to set crop")
    for device in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name):
        video = latest_mp4(device)
        if not video: continue
        crop = DEVICE_TO_CROP.get(device.name)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened(): print("Failed:", video); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / 2)))  # ~2 fps
        total_s = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        win = f"{device.name} | {video.relative_to(root)}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        state = {"p1": None, "crop": crop}
        def click(ev, x, y, *_):
            if ev != cv2.EVENT_LBUTTONDOWN: return
            if state["p1"] is None: state["p1"] = (x, y); return
            x1, y1, x2, y2 = min(state["p1"][0], x), min(state["p1"][1], y), max(state["p1"][0], x), max(state["p1"][1], y)
            if x2 > x1 and y2 > y1:
                state["crop"] = Crop(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
                print(f'DEVICE_TO_CROP["{device.name}"] = {state["crop"]}')
            state["p1"] = None
        cv2.setMouseCallback(win, click)
        while True:
            ok, frame = cap.read()
            if not ok: break
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
            if k == ord("q"): raise SystemExit
            if k == ord("n"): break
            for _ in range(step - 1):
                if not cap.grab(): ok = False; break
            if not ok: break
        cap.release()
        cv2.destroyWindow(win)
    cv2.destroyAllWindows()
