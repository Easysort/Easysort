
import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchreid
import numpy as np
from tqdm import tqdm
import json
from dataclasses import dataclass
from typing import Optional, Dict
import datetime
from easysort.viewer import ImageViewer

model = torchreid.models.build_model(
    name='osnet_x0_75',      # good small ReID model
    num_classes=1000,
    pretrained=True          # use pretrained on ReID datasets
)
model.eval()

preproc = transforms.Compose([
    transforms.Resize((256, 128)),      # common person-ReID size (H, W)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def get_embedding(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    x = preproc(img).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        feat = model(x)            # [1, D]
        feat = F.normalize(feat, dim=1)
    return feat[0]  

@dataclass
class GPTResult:
    person: bool
    person_facing_direction: str
    person_carrying_item: bool
    item_desc: list[str]
    item_cat: list[str]
    item_count: list[int]
    weight_kg: list[float]
    co2_kg: list[float]
    

class ArgoCounter:
    def __init__(self, folder: str):
        self.images = os.listdir(folder)
        self.people_images = sorted([folder + "/" + f for f in self.images if any(x in f for x in ["left", "right", "forward", "back", "unknown"])])
        print(self.people_images[0].split("-")[-4], self.people_images[0].split("-")[-3], self.people_images[0])
        self.people_images = [f for f in self.people_images if f.split("-")[-4] == "11" and int(f.split("-")[-3]) > 24]
        print(*self.people_images[:10], sep="\n")
        print("found", len(self.people_images), "people images")
    
    def check_duplicates(self):
        paths_to_check = list(set(["_".join(f.split("_")[:-2]) for f in self.people_images]))
        meta_paths_to_paths = {p: [] for p in paths_to_check}
        for path in self.people_images:
            path_to_check = "_".join(path.split("_")[:-2])
            meta_paths_to_paths[path_to_check].append(path)
  
        feature_cache = {}
        sorted_meta_paths = sorted(meta_paths_to_paths.items())
        images_to_analyze = [sorted_meta_paths[0][1][0]]
        paths_2_copy = None
        for i in tqdm(range(len(sorted_meta_paths)-1)):
            meta_path_1 = sorted_meta_paths[i][0]
            paths_1 = paths_2_copy if paths_2_copy is not None else sorted_meta_paths[i][1]
            meta_path_2, paths_2 = sorted_meta_paths[i+1]
            paths_2_copy = paths_2.copy()
            if "_".join(meta_path_1.split("_")[:-2]) != "_".join(meta_path_2.split("_")[:-2]): continue
            meta_path_1_idx = int(meta_path_1.split("_")[-2])
            meta_path_2_idx = int(meta_path_2.split("_")[-2])
            if abs(meta_path_1_idx - meta_path_2_idx) > 20: 
                continue
            
            for path_1 in paths_1:
                for path_2 in paths_2:
                    if path_1 not in feature_cache:
                        feature_cache[path_1] = get_embedding(path_1)
                    if path_2 not in feature_cache:
                        feature_cache[path_2] = get_embedding(path_2)
                    
                    feature_1 = feature_cache[path_1]
                    feature_2 = feature_cache[path_2]
                    similarity = F.cosine_similarity(feature_1, feature_2, dim=0)
                    if similarity > 0.75:
                        if path_2 in paths_2_copy:
                            paths_2_copy.remove(path_2)
                        paths_2_copy.append(path_1)
                    else: 
                        images_to_analyze.append(path_2)

                    # save the output with the images side by side and a small text saying the similarity
                    output_path = f"tmp2/{meta_path_1}_{meta_path_2}.jpg".replace("output/", "")
                    img1 = cv2.imread(path_1)
                    img2 = cv2.imread(path_2)
                    # Ensure same number of channels
                    if img1.shape[2] != img2.shape[2]:
                        if img1.shape[2] == 3 and img2.shape[2] == 4:
                            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
                        elif img1.shape[2] == 4 and img2.shape[2] == 3:
                            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
                    h1, w1 = img1.shape[:2]
                    h2, w2 = img2.shape[:2]
                    # Add padding to the smaller image to match heights
                    if h1 != h2:
                        if h1 < h2:
                            pad = h2-h1
                            top = pad // 2
                            bottom = pad - top
                            img1 = cv2.copyMakeBorder(img1, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                        else:
                            pad = h1-h2
                            top = pad // 2
                            bottom = pad - top
                            img2 = cv2.copyMakeBorder(img2, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                    out_img = np.concatenate([img1, img2], axis=1)
                    cv2.putText(out_img, f"Similarity: {similarity:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imwrite(output_path, out_img)


        with open("images_to_analyze.txt", "w") as f:
            for image in images_to_analyze:
                f.write(image + "\n")

    def call_gpt(self):

        GPT_PROMPT = """
        Analyze the image and determine if there is a person, what items they are carrying. The item descriptions should only be items that people are carrying in the image.
        Sometimes people are carrying boxes, bags, etc. If there are items in the bags/boxes, describe them as items too.
        You should state if the person is facing 'left', 'right', 'forward' or 'back'. Make your best guess based on the image.
        You need to give the items a category out of the following list: [Køkkenting, Fritid & Have, Møbler, Boligting, Legetøj, Andet]. In some cases, people are carrying multiple of the same items. There you can use the item_count. Else just say 1.
        Estimate the weight and co2 emission from the item production. If you are a bit unsure, be conservative in your estimates.
        """

        model = "gpt-5-mini-2025-08-07"

        from easysort.gpt_trainer import GPTTrainer
        with open("images_to_analyze.txt", "r") as f:
            images_to_analyze = f.readlines()
        print(len(images_to_analyze), "images to analyze")
        print(images_to_analyze[0])
        images_without_results = [path.strip("\n") for path in images_to_analyze if not os.path.exists(path.strip("\n").replace(".jpg", ".gpt2"))]
        print(len(images_without_results), "images without results")
        unique_images_paths_saved = []
        all_paths_saved = []
        for i in tqdm(range(0, 10000, 500)):
            batch = images_without_results[i:i+500]
            batch_images = [[cv2.imread(image_path)] for image_path in batch]
            gpt_trainer = GPTTrainer(model=model)
            gpt_results = gpt_trainer._openai_call(gpt_trainer.model, GPT_PROMPT, batch_images, GPTResult, max_workers=250)
            for image_path, gpt_result in zip(batch, gpt_results):
                with open(image_path.replace(".jpg", ".gpt2"), "w") as f:
                    f.write(json.dumps(gpt_result.__dict__))
                unique_images_paths_saved.append(image_path)
                all_paths_saved.append(image_path)

        print(len(unique_images_paths_saved), "unique images paths saved")
        print(len(all_paths_saved), "all paths saved")
        print(len(set(all_paths_saved)), "unique paths saved")

    def concat_results(self):
        results: Dict[str, Dict[datetime.datetime, GPTResult]] = {"roskilde": {}, "jyllinge": {}}
        with open("images_to_analyze.txt", "r") as f:
            images_to_analyze = f.readlines()
        for image_path in images_to_analyze:
            with open(image_path.strip("\n").replace(".jpg", ".gpt2"), "r") as f:
                gpt_result = GPTResult(**json.load(f))
                string = image_path.replace("_", "-")
                year, month, day, hour, minute, second = string.split("-")[-9], string.split("-")[-8], string.split("-")[-7], string.split("-")[-5][:2], string.split("-")[-5][2:4], string.split("-")[-5][4:]
                timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                if timestamp not in results[string.split("-")[2].lower()]:
                    results[string.split("-")[2].lower()][timestamp] = []
                results[string.split("-")[2].lower()][timestamp].append((gpt_result, image_path))

        locations = sorted(results.keys())
        for location in locations:
            timestamps = sorted(results[location].keys())
            print("------ ", location, " ------")
            all_objects = []
            all_co2 = 0
            all_weight = 0
            for timestamp in timestamps:
                gpt_result = results[location][timestamp]
                all_objects += sum(gpt_result.item_count)
                all_co2 += sum(gpt_result.estimated_co2_emission_from_item_production_kg)
                all_weight += sum(gpt_result.estimated_weight_of_item_kg)

            # All the objects collected, All the co2 saved, the wieght of all objects, who takes objects

            # Number of objects and weight for each category

            # Co2 estimate * ~0.7-0.75, normal estimate, estimate * 1.3-1.45

            # Objects per day, objects per hour


    def show(self):
        results: Dict[str, Dict[datetime.datetime, GPTResult]] = {"roskilde": {}, "jyllinge": {}}
        with open("/home/lucas/Easysort/images_to_analyze.txt", "r") as f:
            images_to_analyze = f.readlines()
        images_to_analyze = [image_path.strip("\n") for image_path in images_to_analyze]
        print(len(images_to_analyze), "images to analyze")
        print("number of gpt2 files: ", len([f for f in os.listdir("/home/lucas/Easysort/output") if f.endswith(".gpt2")]))
        print("number of relevant gpt2 files: ", len([f for f in os.listdir("/home/lucas/Easysort/output") if f.endswith(".gpt2") and "output/" + f.replace(".gpt2", ".jpg") in images_to_analyze]))
        for path in os.listdir("/home/lucas/Easysort/output"):
            if not path.endswith(".gpt2"):
                continue
            image_path = "output/" + path.replace(".gpt2", ".jpg")
            if image_path not in images_to_analyze:
                continue
            with open("output/" + path, "r") as f:
                gpt_result = GPTResult(**json.load(f))
                string = image_path.replace("_", "-")
                year, month, day, hour, minute, second = string.split("-")[-9], string.split("-")[-8], string.split("-")[-7], string.split("-")[-5][:2], string.split("-")[-5][2:4], string.split("-")[-5][4:]
                timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                if timestamp not in results[string.split("-")[2].lower()]:
                    results[string.split("-")[2].lower()][timestamp] = []
                results[string.split("-")[2].lower()][timestamp].append((gpt_result, image_path))
        
        roskilde_images = [len(results["roskilde"][timestamp]) for timestamp in results["roskilde"]]
        jyllinge_images = [len(results["jyllinge"][timestamp]) for timestamp in results["jyllinge"]]
        print("total roskilde images: ", sum(roskilde_images))
        print("total jyllinge images: ", sum(jyllinge_images))
        print("total images: ", sum(roskilde_images) + sum(jyllinge_images))
        paths_with_objects = []
        paths_without_objects = []
        for location in results:
            for timestamp in results[location]:
                for gpt_result, image_path in results[location][timestamp]:
                    if sum(gpt_result.item_count) > 0:
                        paths_with_objects.append(image_path.strip("\n"))
                    else:
                        paths_without_objects.append(image_path.strip("\n"))

        print(len(paths_with_objects), "images with objects")
        print(len(paths_without_objects), "images without objects")

        # Then show a subset of images without objects
        # ImageViewer(paths_without_objects[:1000])



if __name__ == "__main__":
    counter = ArgoCounter(folder="output")
    # counter.check_duplicates()
    counter.call_gpt()
    # counter.show()