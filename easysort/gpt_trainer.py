
import openai
from easysort.helpers import OPENAI_API_KEY
from typing import List
from dataclasses import dataclass
import json
from pathlib import Path
import base64
from ultralytics import YOLO
import numpy as np

class GPTTrainer:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.openai_client.models.list() # validate api key

    def _openai_call(self, model: str, prompt: str, image_paths: List[str], output_schema: dataclass) -> dataclass:
        images_b64 = [base64.b64encode(Path(image_path).read_bytes()).decode("utf-8") for image_path in image_paths]
        full_prompt = f"{prompt} Return only a json with the following keys and types: {output_schema.__annotations__}"
        content = [{"type": "text", "text": full_prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}} for image_b64 in images_b64]
        response = self.openai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], response_format={"type": "json_object"}, timeout=30,)
        return output_schema(**json.loads(response.choices[0].message.content))
    
class YoloTrainer:
    def __init__(self):
        self.model = YOLO("yolov8s.pt")

    def _is_person_in_image(self, image_paths: List[np.ndarray]) -> List[str]:
        results = self.model(image_paths)
        return [int((result.boxes.cls.cpu().numpy() == 0).sum()) for result in results if result.boxes is not None]


if __name__ == "__main__":
    pass
    
