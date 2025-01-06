import os
import cv2
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np

# Path to image
img_path = "/Users/lucasvilsen/Documents/Documents/EasySort/_old/data/new/d_2024-08-14_0/frame_0154.jpg"

# Read image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

# Define some example boxes and classes
boxes = [
    [202, 571, 679, 888],  # Example box for a plastic bottle
    [1090, 990, 1458, 1199],  # Example box for cardboard
    [48, 0, 721, 653],   # Example box for soft plastic packaging
    [700, 0, 955, 740],   # Example box for soft plastic packaging
    [843, 184, 1329, 864],
    [957, 278, 1503, 977],
    [1011, 0, 1526, 312]
]

class_ids = [0, 0, 1, 1, 2, 3, 4]  # Corresponding class IDs

class_names = {
    0: "metal-can",
    1: "packaging-cardboard", 
    2: "packaging-soft-plastics_occluded",
    3: "packaging-soft-plastics",
    4: "packaging-mixed-plastics"
}

# Create supervision Detections object
boxes_array = np.array(boxes)
class_ids_array = np.array(class_ids)

detections = sv.Detections(
    xyxy=boxes_array,
    class_id=class_ids_array.astype(int)
)

# Annotate image
box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.default())
label_annotator = sv.LabelAnnotator(text_scale=1)  # Reduced text scale from default 1.0
annotated_image = box_annotator.annotate(image.copy(), detections)
annotated_image = label_annotator.annotate(annotated_image, detections, labels=[class_names[id] for id in class_ids])

# Plot with adjusted figure size to prevent zooming
plt.figure(figsize=(W/200, H/200))  # Scale figure size based on image dimensions
plt.imshow(annotated_image)
# plt.title("Example annotations on buf1.webp")
plt.axis('off')
plt.show()
