

import cv2
import numpy as np
import supervision as sv
import torch
from easysort.sorting.sam_utils.predictor import SamPredictor
from easysort.sorting.sam_utils.automatic_mask_generator import SamAutomaticMaskGenerator
from easysort.sorting.sam_utils.build_sam import sam_model_registry
from easysort.sorting.infer_yolov8_ultralytics import Classifier


# !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights

class Segmentation:
    def __init__(self, model_path: str):
        self.model_path = model_path
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

        self.mask_predictor = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)

    def __call__(self, image_path: str, box: sv.BoxAnnotator):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image_rgb)

        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=True
        )

        return masks, scores, logits
    
    def visualize(self, image_path: str, box: np.ndarray, class_id: int):

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        mask_annotator = sv.MaskAnnotator(color=sv.Color.RED)

        masks, _, _ = self(image_path, box)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]
        detections.class_id = [class_id] * len(detections)
        source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        sv.plot_images_grid(
            images=[source_image, segmented_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )

if __name__ == "__main__":
    segmentation = Segmentation(model_path="sam_vit_h_4b8939.pth")
    SOURCE_IMAGE_PATH = "/Users/lucasvilsen/Documents/Documents/EasySort/__old__/_old/test.jpg"
    image = cv2.imread(SOURCE_IMAGE_PATH)
    classifier = Classifier()
    detections = classifier(image)
    box = detections[0].boxes.xyxy.unsqueeze(0).numpy()
    print(detections[0].boxes)
    segmentation.visualize(SOURCE_IMAGE_PATH, box, detections[0].boxes.cls.item())
    annotated_image = image.copy()