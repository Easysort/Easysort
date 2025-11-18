from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry

from tqdm import tqdm
from ultralytics.engine.results import Results
import datetime
from easysort.helpers import Sort
import os
from pathlib import Path
from easysort.helpers import DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH
import json


if __name__ == "__main__":
    # DataRegistry.SYNC()
    yolo_model = "yolov8s.pt"
    yolo_trainer = YoloTrainer(yolo_model)
    gpt_trainer = GPTTrainer()
    gpt_model = gpt_trainer.model
    yolo_person_cls_idx = 0
    project = "argo-people"
    all_paths = DataRegistry.LIST("argo")
    data = list(Sort.since(all_paths, datetime.datetime(2025, 11, 10)))
    data = list(Sort.before(data, datetime.datetime(2025, 11, 17)))
    print(f"Processing {len(data)} paths out of {len(all_paths)}")

    number_of_frames = 0
    total_frames = 0
    
    # Use a set for O(1) lookups instead of O(n) list lookups
    paths_with_people = set()
    if os.path.exists("paths_with_people.txt"):
        with open("paths_with_people.txt", "r") as f:
            paths_with_people = set(line.strip() for line in f if line.strip())
    
    new_paths_with_people = []  # Track new paths to append incrementally
    
    for i, path in enumerate(tqdm(data, desc="Processing paths")):
        # Simplify path construction
        result_path = Path(path.replace(DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH)).with_suffix("")
        files_dir = result_path / yolo_model / project
        
        # Check if directory exists before globbing (saves time)
        if not files_dir.exists():
            continue
        
        # Check if this folder has already been checked (any file in folder is in paths_with_people)
        files_dir_str = str(files_dir) + "/"
        if any(existing_path.startswith(files_dir_str) for existing_path in paths_with_people):
            continue
        
        # Use os.listdir + filter instead of glob - can be faster
        try:
            all_files = os.listdir(files_dir)
            files = [files_dir / f for f in all_files if f.endswith('.json')]
        except OSError:
            continue
        
        for file in files:
            file_str = str(file)  # Convert to string once
            if file_str in paths_with_people:
                continue
            
            # Use context manager for file operations
            try:
                with open(file, "r") as f:
                    file_data = json.load(f)
                if len(file_data) > 0:
                    number_of_frames += 1
                    paths_with_people.add(file_str)  # Add to set
                    new_paths_with_people.append(file_str)  # Track for writing
            except (json.JSONDecodeError, IOError) as e:
                # Handle corrupted files gracefully
                print(f"Error reading {file}: {e}")
                continue
            
            total_frames += 1

        # Append incrementally instead of rewriting entire file
        if i % 10 == 0 and i > 0 and new_paths_with_people:
            with open("paths_with_people.txt", "a") as f:  # Append mode
                f.write("\n" + "\n".join(new_paths_with_people))
            new_paths_with_people.clear()  # Clear after writing
    
    # Write any remaining new paths
    if new_paths_with_people:
        with open("paths_with_people.txt", "a") as f:
            f.write("\n" + "\n".join(new_paths_with_people))

    print(f"Number of frames: {number_of_frames} / {total_frames}")
    print(f"Percentage: {number_of_frames / total_frames * 100}%")

