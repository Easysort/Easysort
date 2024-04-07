from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

data_yaml = "/Users/lucasvilsen/Desktop/EasySort/vision/recycle.v6i.yolov8/data.yaml"
save_dir = "/Users/lucasvilsen/Desktop/ReUse/vision/runs"

model.train(
    data=data_yaml, 
    verbose= True,
    val = False,
    epochs=50,
    seed = 0,
    project = save_dir
)

metrics = model.val()  # evaluate model performance on the validation set
results = model("/Users/lucasvilsen/Desktop/ReUse/vision/recycle.v6i.yolov8/test/images/2a319efa-pic52_jpg.rf.3731665202ce07c61a72d97142cb8441.jpg")  # predict on an image

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk

path = model.export(format="onnx")  # export the model to ONNX format