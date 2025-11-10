
import openai
from easysort.helpers import OPENAI_API_KEY
from typing import List
from dataclasses import dataclass
import json
from pathlib import Path
import base64
from ultralytics import YOLO
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

class GPTTrainer:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.openai_client.models.list() # validate api key
        self.default_model = "gpt-5-2025-08-07"

    def _openai_call(self, model: str, prompt: str, image_paths: List[List[np.ndarray]], output_schema: dataclass, max_workers: int = 10) -> List[dataclass]:
        def process_single(image_arrays):
            images_b64 = [base64.b64encode(cv2.imencode('.jpg', img_array)[1].tobytes()).decode("utf-8") for img_array in image_arrays]
            full_prompt = f"{prompt} Return only a json with the following keys and types: {output_schema.__annotations__}"
            content = [{"type": "text", "text": full_prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} for img_b64 in images_b64]
            response = self.openai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], response_format={"type": "json_object"}, timeout=30,)
            return output_schema(**json.loads(response.choices[0].message.content))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single, image_paths), total=len(image_paths), desc="OpenAI calls"))
        return results
    
class YoloTrainer:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")

    def _is_person_in_image(self, image_paths: List[np.ndarray]) -> List[str]:
        results = self.model(image_paths)
        return [int((result.boxes.cls.cpu().numpy() == 0).sum()) for result in results if result.boxes is not None]


if __name__ == "__main__":
    pass
    
