from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry

from tqdm import tqdm
from ultralytics.engine.results import Results
import datetime
from easysort.helpers import Sort

## Idea for tracking people:
## Detect people with gpt-trainer
## Find out how many unique people
## Embed image and train classifier/SVM

def unpack_results_into_bboxes(results: Results):
    pass

if __name__ == "__main__":
    DataRegistry.SYNC()
    gpt_trainer = GPTTrainer()
    gpt_model = gpt_trainer.model
    yolo_person_cls_idx = 0
    project = "argo-gpt-counter"
    data = DataRegistry.LIST("argo")
    data = list(Sort.since(data, datetime.datetime(2025, 1, 1)))

    for j,path in enumerate(tqdm(data[:10], desc="Processing paths")):
        if ResultRegistry.EXISTS(path, gpt_model, project): continue
        frames = Sampler.unpack(path, crop="auto")
        sorted_frames= Sort.unique_frames(frames)
        prompt = ""
        for frame in sorted_frames:
            results = gpt_trainer._openai_call(gpt_model, prompt, frame)
            ResultRegistry.POST(path, gpt_model, project, frame, results)