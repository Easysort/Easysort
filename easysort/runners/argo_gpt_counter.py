from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry

from tqdm import tqdm
from ultralytics.engine.results import Results
import datetime
from easysort.helpers import Sort


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

    for j,path in enumerate(tqdm(data[:10], desc="Processing paths")):
        print(ResultRegistry.EXISTS(path, yolo_model, project))