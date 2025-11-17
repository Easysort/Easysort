from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry

from tqdm import tqdm
from typing import List
from ultralytics.engine.results import Results
import os
import datetime
from easysort.helpers import Sort

## Idea for tracking people:
## Detect people with gpt-trainer
## Find out how many unique people
## Embed image and train classifier/SVM

if __name__ == "__main__":
    DataRegistry.SYNC()
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

    for j,path in enumerate(tqdm(data, desc="Processing paths")):
        if ResultRegistry.EXISTS(path, yolo_model, project): continue
        frames = Sampler.unpack(path, crop="auto")
        for i in range(0, len(frames), 32):
            batch = frames[i:i+32]
            results = yolo_trainer.model(batch, verbose=False)
            assert len(batch) == len(results)
            for j, result in enumerate(results):
                if result.boxes is None: continue
                elements = zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())
                bboxes = [[float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)] for box, cls, conf in elements if cls == yolo_person_cls_idx]
                ResultRegistry.POST(path, yolo_model, project, str(i+j), bboxes)