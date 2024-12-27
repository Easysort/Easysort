from ultralytics import YOLO
import os

# Switch to tinygrad

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

base_path = "/Users/lucasvilsen/Desktop/EasySort"
data_yaml = "/Users/lucasvilsen/Desktop/EasySort/data/labelled/27-06-2024.v1i.yolov8/data.yaml"
save_dir = "/Users/lucasvilsen/Desktop/EasySort/runs"

model.train(
    data=data_yaml, 
    verbose= True,
    val = False,
    epochs=50,
    seed = 0,
    project = save_dir
)

metrics = model.val()  # evaluate model performance on the validation set
# results = model("/Users/lucasvilsen/Desktop/ReUse/playground/recycle.v6i.yolov8/test/images/2a319efa-pic52_jpg.rf.3731665202ce07c61a72d97142cb8441.jpg")  # predict on an image

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk

# path = model.export(format="onnx")  # export the model to ONNX format