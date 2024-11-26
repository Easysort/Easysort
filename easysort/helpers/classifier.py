# Need:
# !pip install -q inference-gpu[yolo-world]==0.9.12rc1
# !pip install -q supervision==0.19.0rc3
# From: https://colab.research.google.com/github/roboflow/supervision/blob/develop/docs/notebooks/zero-shot-object-detection-with-yolo-world.ipynb#scrollTo=37CMTxw0jSyH

import cv2
import supervision as sv

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

class Classifier: 
    def __init__(self):
        self.model = YOLOWorld(model_id="yolo_world/l")
        self.classes = ["plastic-bottle", "cardboard-box", "plastic-packaging", "other"]
        self.model.set_classes(self.classes)

    def __call__(self, image):
        results = self.model.infer(image)
        detections = sv.Detections.from_inference(results)
        return self.cam_view_to_world_view(detections)
    
    def cam_view_to_world_view(self, detections):
        # Do computations hehe..
        return detections

if __name__ == "__main__":
    SOURCE_IMAGE_PATH = "easysort/helpers/test.jpg"
    image = cv2.imread(SOURCE_IMAGE_PATH)
    classifier = Classifier()
    detections = classifier(image)
    annotated_image = image.copy()

    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
    sv.plot_image(annotated_image, (10, 10))

