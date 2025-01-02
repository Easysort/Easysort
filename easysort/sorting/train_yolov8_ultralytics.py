from ultralytics import YOLO
import os

model = YOLO("easysort/sorting/yolov8n.pt")  # load a pretrained model (recommended for training)

# base_path = "/Users/lucasvilsen/Documents/Documents/EasySort"
data_yaml = "/Users/lucasvilsen/Documents/Documents/EasySort/easysort/sorting/27-06-2024.v1i.yolov8/data.yaml"
save_dir = "/Users/lucasvilsen/Documents/Documents/EasySort/easysort/sorting"

model.train(
    data=data_yaml, 
    verbose= True,
    val = False,
    epochs=50,
    seed = 0,
    project = save_dir
)

metrics = model.val()  # evaluate model performance on the validation set