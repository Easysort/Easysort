import openai, time, fcntl
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

class RunnerJob:
    folder: str
    suffix: List[str] = [".mp4"]
    result_type: type
    interval_mins: int = 5

    def process(self, paths: List[Path], runner: Runner) -> List: raise NotImplementedError

class PusherJob:
    folder: str
    def push(self, paths: List[Path], _type: T): raise NotImplementedError

class ContinuousRunner:
    def __init__(self, run_job: RunnerJob, push_job: PusherJob):
        self.run_job = run_job
        self.push_job = push_job
        self.runner = Runner()

    def run(self):
        print(f"Starting ContinuousRunner for {self.run_job.folder} (interval: {self.run_job.interval_mins}min)")
        while True:
            print(f"\n{'='*50}\nScanning for missing results...")
            all_files, missing = Registry.LIST(self.run_job.folder, suffix=self.run_job.suffix, cond=lambda x: not Registry.EXISTS(x, self.run_job.result_type), return_all=True)
            print(f"Found {len(missing)} missing / {len(all_files)} total")
            if missing:
                print("Waiting for VPN lock...")
                with vpn_lock():
                    print("Lock acquired, processing...")
                    self.run_job.process(missing, self.runner)
                    self.push_job.push(missing)
            print(f"Sleeping {self.run_job.interval_mins} minutes...")
            time.sleep(self.run_job.interval_mins * 60)