from typing import List, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import base64
import openai
import numpy as np
import cv2
from tqdm import tqdm
from easysort.helpers import OPENAI_API_KEY

MODELS_DIR = Path(__file__).resolve().parent.parent / "easyprod" / "models"


class GPTTrainer:
  def __init__(self, model: str = "gpt-5-2025-08-07"):
    self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    self.openai_client.models.list()
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


@dataclass
class ModelSchema:
  """Everything needed to train and run a model. Defined in trainers, consumed by easyprod."""
  name: str
  task: str  # "classify" or "detect"
  classes: list[str]
  base_model: str
  weights_path: Path
  imgsz: int = 224
  preprocess: Callable[[np.ndarray], np.ndarray] | None = None


def _load_model(task: str, path: str):
  if task == "detect":
    from ultralytics import RFDETR
    return RFDETR(path)
  from ultralytics import YOLO
  return YOLO(path)


class Trainer:
  def __init__(self, schema: ModelSchema, dataset: Path | None = None):
    self.schema = schema
    self.dataset = dataset
    self._model = None

  @property
  def model(self):
    if self._model is None:
      s = self.schema
      if s.weights_path.exists():
        path = str(s.weights_path)
        print(f"[Trainer/{s.name}] Loading fine-tuned weights: {s.weights_path}")
      else:
        path = s.base_model
        print(f"[Trainer/{s.name}] WARNING: Weights not found at {s.weights_path}, falling back to base model: {s.base_model}")
      self._model = _load_model(s.task, path)
    return self._model

  def train(self, epochs=30, patience=5, batch=32, **kw):
    s = self.schema
    _load_model(s.task, s.base_model).train(
      data=str(self.dataset), epochs=epochs, patience=patience,
      imgsz=s.imgsz, batch=batch, project=f"{s.name}_model",
      name="train", exist_ok=True, verbose=True, plots=True, **kw,
    )

  def eval(self, **kw):
    s = self.schema
    self.model.val(
      data=str(self.dataset), imgsz=s.imgsz,
      project=f"{s.name}_model", name="val", verbose=True, plots=True, **kw,
    )

  def predict(self, images: list[np.ndarray]) -> list:
    prep = self.schema.preprocess
    batch = [prep(img) if prep else img for img in images]
    results = self.model.predict(batch, verbose=False)
    if self.schema.task == "classify":
      return [self.model.names[int(r.probs.top1)] for r in results]
    return results

  def predict_one(self, image: np.ndarray):
    return self.predict([image])[0]
