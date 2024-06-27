# This script runs the best performing model
# It continuously evaluates the camera
# When an object is found
# it returns the type and midpoint.

# NOT DONE

import cv2
import torch
from ultralytics import YOLO
import math
import numpy as np

model = YOLO("/Users/lucasvilsen/Desktop/EasySort/runs/train/weights/best.pt")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = model.to(device)

cap = cv2.VideoCapture(0)
window_name = 'DETR Object Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # Adjust here to get correct projected positions

def calculate_farthest_point(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return (x1+x2)//2, (y1+y2)//2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        names = model.names

        print("-----------------\nDETECTION:")

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(output_frame, names[cls], org, font, fontScale, color, thickness)
            cv2.drawMarker(output_frame, calculate_farthest_point(box), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            print(f"{names[cls]} at {calculate_farthest_point(box)}")

    cv2.imshow(window_name, output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
