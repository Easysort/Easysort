from pathlib import Path
from typing import List, Dict
from uuid import uuid4
import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm

from easysort.registry import RegistryBase
from easysort.helpers import T, DEBUG

class DataLoader:
    def __init__(self, registry: RegistryBase, classes: List[str] = None, destination: Path = None): 
        self.registry = registry
        self.classes = classes
        self.destination = destination or Path(uuid4())
        self.destination.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val']:
            for cls in self.classes:
                (self.destination / split / cls).mkdir(parents=True, exist_ok=True)
        assert self.classes is not None, "Classes are required"

    def print_distribution(self, inf: str):
        print(f"{inf}: Distribution in {self.destination} is:")
        for split in ['train', 'val']:
            for cls in self.classes:
                print(f"  {split}/{cls}: {len(os.listdir(self.destination / split / cls))}")

    def from_yolo_dataset(self, dataset_path: Path): 
        for split in ['train', 'val']:
            for cls in self.classes:
                for file in tqdm(list((dataset_path / split / cls).glob("*.jpg")), desc=f"Copying {split} {cls}"):
                    if (self.destination / split / cls / file.name).exists(): continue
                    shutil.copy(file, self.destination / split / cls / file.name)

        if DEBUG > 0: self.print_distribution(f"From {dataset_path}")

    def sample_image(self) -> np.ndarray:
        for split in ['train', 'val']:
            for cls in self.classes:
                file = random.choice(list((self.destination / split / cls).glob("*.jpg")))
                return cv2.imread(file)
        
