from typing import List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import base64
from pathlib import Path
import openai
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from easysort.helpers import OPENAI_API_KEY
from easysort.dataloader import DataLoader


class GPTTrainer:
    def __init__(self, model: str = "gpt-5-2025-08-07"):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.openai_client.models.list() # breaks if api key is invalid
        self.model = model

    def _openai_call(self, model: str, prompt: str, image_paths: List[List[np.ndarray]], output_schema: dataclass, max_workers: int = 10) -> List[dataclass]:
        def process_single(image_arrays):
            images_b64 = [base64.b64encode(cv2.imencode('.jpg', img_array)[1].tobytes()).decode("utf-8") for img_array in image_arrays]
            full_prompt = f"{prompt} Return only a json with the following keys and types: {output_schema.__annotations__}"
            content = [{"type": "text", "text": full_prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} for img_b64 in images_b64]
            response = self.openai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], response_format={"type": "json_object"}, timeout=90,)
            return output_schema(**json.loads(response.choices[0].message.content))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single, image_paths), total=len(image_paths), desc="OpenAI calls"))
        return results

class YoloTrainer:
    # FIXED 
    def __init__(self, name: str, classes: List[str], model_path: str = "yolo11s-cls.pt", dataloader: DataLoader = None):
        self.name = name
        self.model = YOLO(model_path)
        self.classes = classes
        self.dataset = dataloader.destination

    def train(self, dataset: Path, epochs: int = 30, patience: int = 5, imgsz: int = 224, batch: int = 32):
        self.model.train(
            data=str(dataset),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            project=f"{self.name}_model",
            name="train",
            exist_ok=True,
            verbose=True,
            plots=True,
        )

    def eval(self):
        self.model.val(
            data=str(self.dataset),
            imgsz=224,
            batch=32,
            project=f"{self.name}_model",
            name="val",
            verbose=True,
            plots=True,
        )