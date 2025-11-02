import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
import cv2


@dataclass
class KeyAction:
    key: str
    label: str
    handler: Optional[Callable[[Path], Optional[Dict]]] = None


def run_viewer(name: str, description: str, images: List[Path], actions: List[KeyAction]) -> None:
    imgs = [p for p in images if p.exists() and p.is_file()]
    keymap = {ord(a.key.lower()): a for a in actions} | {ord(a.key.upper()): a for a in actions}
    i, n, win = 0, len(imgs), (name or "Viewer")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while n:
        im = cv2.imread(str(imgs[i]));  h = 28
        if im is None: i = (i + 1) % n; continue
        overlay = [name, description, f"{imgs[i].name}  ({i+1}/{n})", "Keys:"] + [f"  {a.key} = {a.label}" for a in actions] + ["  n = Next, ESC = Quit"]
        y = 26
        for line in overlay:
            cv2.putText(im, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA); y += h
        cv2.imshow(win, im)
        k = cv2.waitKey(0) & 0xFF
        if k == 27: break
        if k in (ord('n'), ord('N')): i = (i + 1) % n; continue
        act = keymap.get(k)
        if act:
            extra = act.handler(imgs[i]) if act.handler else None
            payload = {"image": imgs[i].name, "dir": str(imgs[i].parent), "key": act.key, "label": act.label, "timestamp": datetime.utcnow().isoformat()}
            if isinstance(extra, dict): payload.update(extra)
            with open(imgs[i].with_suffix("").with_name(imgs[i].stem + ".resp.json"), "w") as f: json.dump(payload, f)
            i = (i + 1) % n
    cv2.destroyAllWindows()



