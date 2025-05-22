# Need:
# !pip install -q inference-gpu[yolo-world]==0.9.12rc1
# !pip install -q supervision==0.19.0rc3
# From: https://colab.research.google.com/github/roboflow/supervision/blob/develop/docs/notebooks/zero-shot-object-detection-with-yolo-world.ipynb#scrollTo=37CMTxw0jSyH

import cv2
import supervision as sv
import numpy as np
import torch

from easysort.common.logger import EasySortLogger
from easysort.utils.detections import Detection
from inference.models.yolo_world.yolo_world import YOLOWorld
import time
from easysort.common.timer import TimeIt

LOGGER = EasySortLogger()
RANDOM_IMAGE_TENSOR = torch.rand((980, 1280, 3))

YOLO_WORLD_CLASSES = {
    "Coffee cup lid": "Small plastic lid",
    "Bottle cap": "Plastic bottle cap",
    "PS styrofoam": "Polystyrene foam",
    "Light blue pet bottle": "Light blue PET bottle",
    "Clear pet bottle food covered": "Clear PET food bottle",
    "Clear pet bottle non food convered": "Clear PET non-food bottle",
    "Clear pet bottle": "Transparent PET bottle",
    "Brown pet bottle": "Brown PET bottle",
    "Coloured pet bottle": "Colored PET bottle",
    "Dark blue pet bottle": "Dark blue PET bottle",
    "Green pet bottle": "Green PET bottle",
    "White pet bottle": "White PET bottle",
    "Pet bottle other colour": "Colored PET bottle",
    "Coloured hdpe bottle": "Colored HDPE bottle",
    "Coloured yoghurt plastic bottle": "Colored yogurt bottle",
    "Natural hdpe bottle": "Natural HDPE bottle",
    "Natural yoghurt plastic bottle": "Natural yogurt bottle",
    "Black container": "Black plastic container",
    "Black plastic bottle": "Black plastic bottle",
    "Other black plastic": "Black plastic items",
    "Clear container": "Clear plastic container",
    "Coloured container": "Colored plastic container",
    "White container": "White plastic container",
    "Other plastic": "Miscellaneous plastics",
    "Bottle label film": "Labeling plastic film",
    "Film bubble wrap": "Bubble wrap film",
    "Clear film": "Transparent plastic film",
    "Clear printed film": "Printed clear film",
    "Clear plastic packaging": "Transparent plastic packaging",
    "Coloured film": "Colored plastic film",
    "Filled bag": "Filled plastic bag",
    "Coloured printed film": "Colored printed film",
    "Metallised film": "Metalized plastic film",
    "Black film": "Black plastic film",
    "Aluminium Aerosol": "Aluminum aerosol cans",
    "Aluminium cans": "Aluminum beverage cans",
    "Aluminium other": "Aluminum materials",
    "Steel cans": "Steel cans",
    "Metal other": "Miscellaneous metals",
    "Cardboard packaging": "Cardboard boxes",
    "Coloured cardboard packaging": "Colored cardboard boxes",
    "Paper packaging": "Paper packaging",
    "Clean paper sheet (Can be written on, but no food)": "Clean paper sheets",
    "Dirty paper (Food, etc.)": "Contaminated paper",
    "Wipes": "Disposable wipes",
    "Tetra Pak Carton (Milk, juice cartons, etc.)": "Tetra Pak cartons",
    "Cup drink (Disposable cups)": "Disposable drink cups",
    "Cup food (Disposable food containers)": "Disposable food containers",
    "Weee (Waste Electrical and Electronic Equipment)": "Electronic waste",
    "Batteries": "Batteries",
    "Clear glass bottle": "Clear glass bottles",
    "Green glass bottle": "Green glass bottles",
    "Brown glass bottle": "Brown glass bottles",
    "Clear glass cullet (broken)": "Broken clear glass",
    "Green glass cullet (broken)": "Broken green glass",
    "Brown glass cullet (broken)": "Broken brown glass",
    "Glass with plastic": "Glass mixed with plastic",
    "Other glass": "Miscellaneous glass",
    "Clothes/Fabric": "Clothing and textiles",
    "Fine (small granular material)": "Fine granular waste",
    "Sanitary": "Sanitary waste",
    "Other (General waste)": "General miscellaneous waste",
}

test_yoloworld_classes = [
    "plastic bottle",
    "coloured plastic bottle",
    "soda can",
    "paper_document",
    "plastic_wrap",
    "snack_box",
    "cardboard_box",
    "plastic_lid",
]


class ClassifierYoloWorld:
    def __init__(self, classes: list[str]):
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.classes = classes
        self.model.set_classes(self.classes)

    @TimeIt("Bbox classification")
    def __call__(self, image):
        timestamp = time.time()
        results = self.model.infer(image)
        detections = []
        for prediction in results.predictions:
            bbox = np.array([
                prediction.x - prediction.width / 2,
                prediction.y + prediction.height / 2,
                prediction.x + prediction.width / 2,
                prediction.y - prediction.height / 2,
            ])
            detections.append(
                Detection(
                    box=bbox,
                    class_id=prediction.class_id,
                    names={str(prediction.class_id): prediction.class_name},
                    confidence=prediction.confidence,
                    timestamp=timestamp,
                )
            )
        return detections

    def test_speed(self) -> None:
        time0 = time.time()
        self(RANDOM_IMAGE_TENSOR)
        LOGGER.info(f"Time taken: {round(time.time() - time0, 2)} seconds")

    def visualize(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        detections = self(image)
        sv.plot_image(sv.BoundingBoxAnnotator(thickness=2).annotate(image, detections), (10, 10))

    def cam_view_to_world_view(self, detections):
        # Do computations...
        return detections


if __name__ == "__main__":
    SOURCE_IMAGE_PATH = "__old__/_old/test.jpg"
    image = cv2.imread(SOURCE_IMAGE_PATH)
    classifier = ClassifierYoloWorld(classes=test_yoloworld_classes)
    detections = classifier(image)
    print(detections)

    # annotated_image = image.copy()
    # BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    # LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
    # annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    # annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
    # sv.plot_image(annotated_image, (10, 10))
