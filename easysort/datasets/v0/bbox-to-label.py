from PIL import Image
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import requests
import base64
import io
import json

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "easysort/datasets/v0/weights/sam_vit_h_4b8939.pth"

GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent'
GEMINI_API_KEY = 'AIzaSyD_xKSKNk-n2XU85KbEnMmj6tEJGFXVFxA'

SAVE_DIR = 'easysort/datasets/v0/images'

PROMPT = """
Based on this image, give an estimate of the purity level and waste fraction of the item.
The purity level should be from 0 (extremely dirty) to 10 (completely clean)
Waste fraction should be on of: Hard Plastics, Flexible plastics, Metals, Paper, Cardboard, Composite, Electronics, Glass, Clothers, Disposible waste, Highly mixed waste, Other
Return in json format with the following keys: purity (int), waste fraction (str)
"""

waste_fractions = ['hard plastics', 'flexible plastics', 'metals', 'paper', 'cardboard', 'composite', 'electronics', 'glass', 'clothers', 'disposible waste', 'highly mixed waste', 'other']

@dataclass
class Detection:
    bbox: np.ndarray
    purity: int
    waste_fraction: str
    subfraction: str

    def __post_init__(self):
        if self.waste_fraction.lower() not in waste_fractions:
            raise ValueError(f"Invalid waste fraction: {self.waste_fraction}")
        assert 0 <= self.purity <= 10, f"Invalid purity: {self.purity}"

def prepare_image_for_training(image_path: str, mask_generator: SamAutomaticMaskGenerator) -> None:
    image = get_image(image_path)
    print(f"Getting detections for image {image_path}")
    detections = get_sam2_detections(image, mask_generator)
    print(f"Detections: {detections}")
    image_crops = get_image_crops(image, detections)
    if len(image_crops) == 0:
        print(f"No detections found for image {image_path}")
        return None
    print(f"Generating labels for {len(image_crops)} crops")
    image_crops = image_crops[:1]
    crops_and_detections = [(image_crop, generate_label_from_crop(image_crop, xyxy)) for image_crop, xyxy in image_crops]
    print(f"Saving {len(crops_and_detections)} crops")
    save_image_crops(crops_and_detections)
    print(f"Saved {len(crops_and_detections)} crops")

def save_image_crops(crops_and_detections: List[Tuple[np.ndarray, Detection]]) -> None:
    for i, (image_crop, detection) in enumerate(crops_and_detections):
        image_crop_with_label = add_label_to_image(image_crop, detection)
        # Add .png extension and ensure directory exists
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f"crop_{i}.png")
        cv2.imwrite(save_path, image_crop_with_label)

def add_label_to_image(image_crop: np.ndarray, detection: Detection) -> np.ndarray:
    image_crop_with_label = image_crop.copy()
    image_crop_with_label = cv2.putText(image_crop_with_label, f"{detection.waste_fraction} {detection.purity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image_crop_with_label

def get_image_crops(image: np.ndarray, detections: sv.Detections) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(detections.xyxy) == 0:
        return []
    image_crops = []
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = xyxy
        image_crop = image[y1:y2, x1:x2]
        image_crops.append((image_crop, xyxy))
    return image_crops

def generate_label_from_crop(image_crop: np.ndarray, xyxy: np.ndarray) -> Detection:
    # image_pil = Image.fromarray(image_crop)
    # image_bytes = io.BytesIO()
    # image_pil.save(image_bytes, format='PNG')
    # image_bytes = image_bytes.getvalue()
    # image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # payload = {
    #     "contents": [{
    #         "parts": [
    #             {"text": PROMPT},
    #             {
    #                 "inline_data": {
    #                     "mime_type": "image/png",
    #                     "data": image_b64
    #                 }
    #             }
    #         ]
    #     }]
    # }

    # headers = {
    #     "Content-Type": "application/json",
    #     "x-goog-api-key": GEMINI_API_KEY
    # }
    
    # response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    
    # if response.status_code != 200:
    #     raise Exception(f"API request failed with status code {response.status_code}")

    # result = response.json()
    # text_response = result['candidates'][0]['content']['parts'][0]['text']
    # waste_fraction, purity = parse_gemini_response(text_response)
    waste_fraction = 'hard plastics'
    purity = 10
    subfraction = 'hard plastics'
    
    return Detection(
        bbox=xyxy,
        waste_fraction=waste_fraction,
        purity=int(purity),
        subfraction=subfraction
    )

def parse_gemini_response(response: str) -> Tuple[str, int]:
    # Extract the JSON string from the response
    json_str = response.strip().split("```json\n")[1].split("\n```")[0]
    response_json = json.loads(json_str)
    waste_fraction = response_json["waste fraction"]
    purity = response_json["purity"]
    return waste_fraction, purity

def get_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def get_sam2_detections(image: np.ndarray, mask_generator: SamAutomaticMaskGenerator) -> List[np.ndarray]:
    sam_result = mask_generator.generate(image)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    if detections.class_id is None:
        detections.class_id = np.zeros(len(detections), dtype=int)
    return detections

def prep_model():
    print(f"Loading model {MODEL_TYPE} from {CHECKPOINT_PATH}")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        min_mask_region_area=1000,
        points_per_side=100,
        pred_iou_thresh=0.99,
        stability_score_thresh=0.99
    )
    print(f"Model {MODEL_TYPE} loaded")
    return mask_generator

if __name__ == "__main__":
    mask_generator = prep_model()
    image_path = "easysort/datasets/v0/data/1.png"
    print(f"Image {image_path} exists: {os.path.exists(image_path)}")
    prepare_image_for_training(image_path, mask_generator)


