import os
import cv2
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np

# Index to choose which image/label pair to display
IMAGE_INDEX = 15

# Path to dataset
dataset_path = "/Users/lucasvilsen/Documents/Documents/EasySort/_old/data/labelled/27-06-2024.v1i.yolov8/train"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# Get list of files
image_files = sorted(os.listdir(images_path))
label_files = sorted(os.listdir(labels_path))

# Read image and corresponding label
img_path = os.path.join(images_path, image_files[IMAGE_INDEX])
label_path = os.path.join(labels_path, label_files[IMAGE_INDEX])

# Read image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

# Read YOLO format labels and convert to supervision Detections
boxes = []
class_ids = []
class_names = {
    0: "bottle-plastic",
    1: "cardboard", 
    3: "packaging-soft-plastic"
}

with open(label_path, 'r') as f:
    for line in f.readlines():
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - width/2) * W)
        y1 = int((y_center - height/2) * H)
        x2 = int((x_center + width/2) * W)
        y2 = int((y_center + height/2) * H)
        boxes.append([x1, y1, x2, y2])
        class_ids.append(int(class_id))  # Convert class_id to integer

# Create supervision Detections object with empty arrays if no boxes found
boxes_array = np.array(boxes) if boxes else np.zeros((0, 4))
class_ids_array = np.array(class_ids) if class_ids else np.zeros(0)

detections = sv.Detections(
    xyxy=boxes_array,
    class_id=class_ids_array.astype(int)  # Ensure class IDs are integers
)

# Annotate image
box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.default())
label_annotator = sv.LabelAnnotator()
annotated_image = box_annotator.annotate(image.copy(), detections)
annotated_image = label_annotator.annotate(annotated_image, detections, labels=[class_names[id] for id in class_ids])

# Plot
plt.figure(figsize=(10,10))
plt.imshow(annotated_image)
plt.title(f"Image {IMAGE_INDEX}: {image_files[IMAGE_INDEX]}")
plt.axis('off')
plt.show()