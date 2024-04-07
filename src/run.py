# This script runs the best performing model
# It continuously evaluates the camera
# When an object is found
# it returns the type and midpoint.

# NOT DONE

import cv2
import torch
from ultralytics import YOLO
import math

model = YOLO("/Users/lucasvilsen/Desktop/EasySort/vision/runs/train/weights/best.pt")

# Set the device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = model.to(device)

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    results = model(frame, stream=True)

    # Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        names = model.names
        # result.show()  # display to screen

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, names[cls], org, font, fontScale, color, thickness)

        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        # result.show()  # display to screen

    # Display the resulting frame
    cv2.imshow('DETR Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
