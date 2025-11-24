import cv2
from pathlib import Path
from typing import List
import random
from PIL import Image
from tqdm import tqdm
from easysort.sampler import Sampler

def label_folder(folder: str, suffix: str = ".label"):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]
    cv2.namedWindow("Label: 0-9, q=quit", cv2.WINDOW_NORMAL)
    for img_path in images:
        label_path = img_path.with_suffix(suffix)
        if label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imshow("Label: 0-9, q=quit", img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if ord('0') <= key <= ord('9'):
            label_path.write_text(str(key - ord('0')))
    cv2.destroyAllWindows()

def unpack_and_save(paths: List[str]):
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    for _ in tqdm(range(10)):
        path = random.choice(paths)
        frames = Sampler.unpack(path, crop="auto")
        for frame_idx, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f"{output_dir}/{path.split('/')[-1]}_{frame_idx}.jpg")



if __name__ == "__main__":
    folder = "tmp"
    label_folder(folder)
