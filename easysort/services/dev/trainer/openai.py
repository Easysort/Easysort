# The main idea here is to have a general trainer that allows to give a dataset
# with a prompt and images, specified json output format and a model, and it will
# use OpenAI inference to gain a first hand idea of the performance of the model,
# create a 

from easysort.services.dev.trainer.viewer import run_viewer, KeyAction
from easysort.common.environment import Env

import openai
from typing import List, Union
from dataclasses import dataclass, fields
import base64
from pathlib import Path
import json

@dataclass
class Dataset:
    images: List[Path]
    labels: Path
    creation_description: str
    creation_actions: List[KeyAction]


@dataclass
class TrainerConfig:
    models: List[str]
    prompt: List[str]
    expected_output_schema: dataclass


class OpenAITrainer:

    """
    Auto creates return json schema based on the expected_output_schema and inserts into the prompt

    A simple way to check image format on pretrained models to get large scale data for simple, tinygrad models.
    Perform small human validation to check pretrained model performance.
    """

    def __init__(self):
        self.client = openai.OpenAI(api_key=Env.OPENAI_API_KEY)

    @property
    def system_prompt(self) -> str: 
        return "Return a json with the following keys and types: " + ", ".join([f"{field.name}: {field.type}" for field in fields(self.trainer_config.expected_output_schema)])

    def _openai_call(self, model: str, prompt: str, images: List[str], response_format: Union[str, dict, None] = "json_object") -> str:
        content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}} for image_b64 in images]
        # Normalize response_format: allow "json_object" string, dict, or ignore unsupported types
        if isinstance(response_format, str):
            rf = {"type": response_format} if response_format else {"type": "json_object"}
        elif isinstance(response_format, dict):
            rf = response_format
        else:
            rf = {"type": "json_object"}
        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], response_format=rf, timeout=30,)
        return response.choices[0].message.content

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # def _enforce_json_output(self, response: str) -> dict:
    #     self.trainer_config.expected_output_schema(**json.loads(response))

    # def _call(self, model: str, prompt: str, images: List[str]) -> str:
    #     images_b64 = [self._image_to_base64(image_path) for image_path in images]
    #     response = self._openai_call(model, prompt, images_b64)
    #     return self._enforce_json_output(response)



    # def check_labels(self, dataset: Dataset) -> None:
    #     if not dataset.labels.exists(): run_viewer("Create Labels", dataset.creation_description, dataset.images, dataset.creation_actions)

    # def concat_datasets(self) -> None:
    #     # datasets = [im for dataset in self.trainer_config.datasets for im in dataset.images]
    #     pass

    # def train(self, image_loader: Optional[Callable[[str], Image.Image]] = None) -> None:
    #     if image_loader is None: image_loader = self.load_image
    #     for dataset in self.trainer_config.datasets: self.check_labels(dataset)
    #     if self.trainer_config.concat_datasets: self.concat_datasets()
    #     print("All datasets look healthy, starting training...")


