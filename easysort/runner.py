import openai, time, fcntl, functools
from easysort.helpers import OPENAI_API_KEY, T
from typing import List, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import base64
from ultralytics import YOLO
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from easysort.sampler import Crop, Sampler
from easysort import Registry
import cv2

VPN_LOCK_FILE = Path("/tmp/verdis_vpn.lock")

@contextmanager
def vpn_lock(): # TODO: Make a better solution to this lock
    """Acquire exclusive lock - blocks if VPN is active. Ensures OpenAI calls don't go through VPN."""
    VPN_LOCK_FILE.touch(exist_ok=True)
    with open(VPN_LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class Runner:
    def __init__(self, model: str = "gpt-5-mini-2025-08-07"):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.openai_client.models.list() # breaks if api key is invalid
        self.model = model
    
    def gpt(self, videos_missing_results: List[List[np.ndarray]], output_schema: T, task_prompt: str = "", model: str = "", max_workers: int = 10) -> List[T]:
        schema = {k: v for k, v in output_schema.__annotations__.items() if k not in ("id", "metadata")}
        model = model or self.model
        def process_single(image_arrays):
            images_b64 = [base64.b64encode(cv2.imencode('.jpg', img_array)[1].tobytes()).decode("utf-8") for img_array in image_arrays]
            full_prompt = f"{task_prompt}\nReturn only a json with the following keys and types: {schema}"
            content = [{"type": "text", "text": full_prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} for img_b64 in images_b64]
            response = self.openai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], response_format={"type": "json_object"}, timeout=90,)
            return output_schema(**json.loads(response.choices[0].message.content))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single, videos_missing_results), total=len(videos_missing_results), desc="OpenAI calls"))
        return results
    
    def yolo(self, videos_missing_results: List[List[np.ndarray]], crop_index_func: Callable[[Path], Crop], classes: List[int], model_path: str = "yolov8m.pt"):
        # for video_path in videos_missing_results:
        #     crop = crop_index_func(video_path)
        #     frames = Sampler.unpack(video_path, crop=crop)
        #     results = self.model(frames, classes=classes)



        # model = YOLO(model_path)
        pass


@dataclass(frozen=True, slots=True)
class PersonCrop:
    frame_idx: int
    direction: str
    image: np.ndarray


def box_inside_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    return 0.0 if not a_area else (iw * ih) / a_area


def facing_direction(kxy: np.ndarray, kconf: np.ndarray, face_conf: float = 0.4, dx_forward: float = 10) -> str:
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
    vis = lambda i, t=face_conf: float(kconf[i]) > t
    if not (vis(L_SH, 0.2) and vis(R_SH, 0.2)): return "unknown"
    sx = float(kxy[L_SH, 0] + kxy[R_SH, 0]) / 2
    if not (vis(NOSE) or vis(L_EYE) or vis(R_EYE) or vis(L_EAR) or vis(R_EAR)): return "back"
    if not vis(NOSE):
        if vis(L_EAR) and not vis(R_EAR): return "right"
        if vis(R_EAR) and not vis(L_EAR): return "left"
        return "unknown"
    dx = float(kxy[NOSE, 0]) - sx
    if abs(dx) <= dx_forward: return "forward"
    return "right" if dx > 0 else "left"


def side_view_visible(kconf: np.ndarray, conf: float = 0.4) -> bool:
    L_SH, R_SH, L_K, R_K, L_A, R_A = 5, 6, 13, 14, 15, 16
    vis = lambda i: float(kconf[i]) > conf
    return (vis(L_SH) or vis(R_SH)) and (vis(L_K) or vis(R_K) or vis(L_A) or vis(R_A))


@functools.cache
def _yolo(model_path: str) -> YOLO:
    return YOLO(model_path)


def extract_person_crops(
    frames: list[np.ndarray],
    crop: Crop | None = None,
    *,
    model_path: str = "yolo11n-pose.pt",
    batch: int = 16,
    min_w: int = 80,
    min_h: int = 200,
    min_in_crop: float = 0.5,
    pad: float = 0.08,
) -> list[PersonCrop]:
    if not frames: return []
    crop_box = (crop.x, crop.y, crop.x + crop.w, crop.y + crop.h) if crop else None
    model, out = _yolo(model_path), []
    for s in range(0, len(frames), batch):
        res = model(frames[s:s + batch], classes=[0], verbose=False)
        for bi, r in enumerate(res):
            if r.boxes is None or not len(r.boxes): continue
            kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            kconf = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else None
            for j, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.cpu().numpy().astype(int)):
                if x2 - x1 < min_w or y2 - y1 < min_h: continue
                if crop_box and box_inside_ratio((x1, y1, x2, y2), crop_box) < min_in_crop: continue
                if kconf is not None and not side_view_visible(kconf[j]): continue
                d = facing_direction(kxy[j], kconf[j]) if kconf is not None else "unknown"
                im = frames[s + bi]
                h, w = im.shape[:2]
                px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
                cx1, cy1, cx2, cy2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
                if cx2 > cx1 and cy2 > cy1: out.append(PersonCrop(s + bi, d, im[cy1:cy2, cx1:cx2]))
    return out


def extract_person_crops_from_video(
    video_path: Path | str,
    crop: Crop | None = None,
    *,
    model_path: str = "yolo11n-pose.pt",
    batch: int = 16,
    min_w: int = 80,
    min_h: int = 200,
    pad: float = 0.08,
) -> list[PersonCrop]:
    path = Registry._registry_path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return []
    out, frames, idxs, i = [], [], [], 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if crop is not None: frame = frame[crop.y:crop.y + crop.h, crop.x:crop.x + crop.w]
            frames.append(frame); idxs.append(i); i += 1
            if len(frames) < batch: continue
            out += [PersonCrop(idxs[c.frame_idx], c.direction, c.image) for c in extract_person_crops(frames, None, model_path=model_path, batch=batch, min_w=min_w, min_h=min_h, min_in_crop=0.0, pad=pad)]
            frames.clear(); idxs.clear()
        if frames:
            out += [PersonCrop(idxs[c.frame_idx], c.direction, c.image) for c in extract_person_crops(frames, None, model_path=model_path, batch=len(frames), min_w=min_w, min_h=min_h, min_in_crop=0.0, pad=pad)]
    finally:
        cap.release()
    return out


@functools.cache
def _reid():
    import torch, torchreid
    from torchvision import transforms
    model = torchreid.models.build_model(name="osnet_x0_75", num_classes=1000, pretrained=True).eval()
    pre = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 128)), transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return model, pre


def reid_embeddings(images: list[np.ndarray]):
    import torch
    import torch.nn.functional as F
    model, pre = _reid()
    if not images: return torch.empty((0, 0))
    x = torch.stack([pre(im[:, :, ::-1]) for im in images])  # BGR->RGB
    with torch.no_grad(): return F.normalize(model(x), dim=1)


def dedupe_reid(crops: list[PersonCrop], window: int = 20, thresh: float = 0.75) -> list[PersonCrop]:
    if len(crops) < 2: return crops
    embs, keep = reid_embeddings([c.image for c in crops]), []
    for i, c in enumerate(crops):
        dup = False
        for j in reversed(keep):
            if c.frame_idx - crops[j].frame_idx > window: break
            if float((embs[i] * embs[j]).sum()) > thresh: dup = True; break
        if not dup: keep.append(i)
    return [crops[i] for i in keep]

class RunnerJob:
    folder: str
    suffix: List[str] = [".mp4"]
    result_type: type
    interval_mins: int = 5

    def process(self, paths: List[Path], runner: Runner) -> List: raise NotImplementedError

class PusherJob:
    folder: str
    def push(self): raise NotImplementedError

class ContinuousRunner:
    def __init__(self, run_job: RunnerJob, push_job: PusherJob):
        self.run_job = run_job
        self.push_job = push_job
        self.runner = Runner()

    def run(self):
        print(f"Starting ContinuousRunner for {self.run_job.folder} (interval: {self.run_job.interval_mins}min)")
        while True:
            print(f"\n{'='*50}\nScanning for missing results...")
            all_files, missing = Registry.LIST(self.run_job.folder, suffix=self.run_job.suffix, return_all=True, check_exists_with_type=self.run_job.result_type)
            print(f"Found {len(missing)} missing / {len(all_files)} total")
            print("Waiting for VPN lock...")
            if missing:
                self.run_job.process(missing, self.runner)
                # Always run the pusher so it can recover from previous push failures.
                self.push_job.push()
            print(f"Sleeping {self.run_job.interval_mins} minutes...")
            time.sleep(self.run_job.interval_mins * 60)