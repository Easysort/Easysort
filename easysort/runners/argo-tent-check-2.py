
import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchreid
import numpy as np
from tqdm import tqdm

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
    

class ArgoCounter:
    def __init__(self, folder: str):
        self.images = os.listdir(folder)
        self.people_images = sorted([folder + "/" + f for f in self.images if any(x in f for x in ["left", "right", "forward", "back", "unknown"])])
        print("found", len(self.people_images), "people images")

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
                    
        
    def remove_duplicates(self):
        pass

    def send_to_gpt(self):
        pass

    def output_results(self):
        pass







if __name__ == "__main__":
    counter = ArgoCounter(folder="output")