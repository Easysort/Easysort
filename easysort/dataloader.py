from pathlib import Path
from typing import List, Dict
from uuid import uuid4
import os
import shutil
import random
import cv2
import numpy as np
from typing import Callable
from tqdm import tqdm

from easysort.registry import RegistryBase
from easysort.helpers import T, DEBUG, registry_file_to_local_file_path

class DataLoader: # TODO: Work for other than .jpg files
    def __init__(self, registry: RegistryBase, classes: List[str] = None, destination: Path = None, force_recreate: bool = False): 
        self.registry = registry
        self.classes = classes
        self.destination = destination or Path(uuid4())
        if force_recreate and self.destination.exists(): shutil.rmtree(self.destination)
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

    def from_registry(self, _label_type: T, _label_json_to_category_func: Callable = None, file_to_saved_img_func: Callable = None, prefix: str = ""):
        """
        file_to_saved_img_func: Function that takes in the original filepath with the type and then does whatever to save the image(s) to the correct destination.
         - This is useful when needed e.g. to take in 3 diff images but only 1 saved label for the category (verdis motion).


        """
        if DEBUG > 0: print(f"Loading data from registry with prefix {prefix} and check_exists_with_type {_label_type}")
        files = self.registry.LIST(prefix = prefix, check_exists_with_type = _label_type)
        if DEBUG > 0: print(f"Found {len(files)} annotated files looking like this: {files[0]}")
        all_dataset_file_names = [(Path(root) / fname).name for root, _, files in os.walk(self.destination) for fname in files]
        missing_files = [file for file in files if registry_file_to_local_file_path(file).name not in all_dataset_file_names]
        print(f"Found {len(missing_files)} missing files out of {len(files)} annotated files")


        for file in tqdm(missing_files, desc="From registry"):
            qa_result = self.registry.GET(file, _label_type)
            category = _label_json_to_category_func(qa_result)
            train_split = "train" if random.random() < 0.8 else "val"
            save_path = self.destination / train_split / str(category)
            save_path.mkdir(parents=True, exist_ok=True)
            if file_to_saved_img_func: file_to_saved_img_func(self.registry, file, save_path)
            else: 
                img = self.registry.GET(file, self.registry.DefaultMarkers.ORIGINAL_MARKER)
                cv2.imwrite(save_path / file.name, img)

        if DEBUG > 0: self.print_distribution(f"From registry {_label_type}")

    def from_yolo_dataset(self, dataset_path: Path, convert_function: Callable = None): 
        for split in ['train', 'val']:
            for cls in self.classes:
                for file in tqdm(list((dataset_path / split / cls).glob("*.jpg")), desc=f"Copying {split} {cls}"):
                    if (self.destination / split / cls / file.name).exists(): continue
                    if not convert_function: shutil.copy(file, self.destination / split / cls / file.name)
                    else:
                        img = cv2.imread(file)
                        img = convert_function(img)
                        cv2.imwrite(self.destination / split / cls / file.name, img)

        if DEBUG > 0: self.print_distribution(f"From {dataset_path}")

    def sample_image(self) -> np.ndarray:
        for split in ['train', 'val']:
            for cls in self.classes:
                file = random.choice(list((self.destination / split / cls).glob("*.jpg")))
                return cv2.imread(file)
        
