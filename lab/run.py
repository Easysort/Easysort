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

model = YOLO("/Users/lucasvilsen/Desktop/EasySort/runs/train4/weights/best.pt")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = model.to(device)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

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

    # You can pass multiple frames, we only pass on, so just taking the first and only result element
    results = model(frame, stream=True)
    result = list(results)[0]
    
    boxes = list(result.boxes)
    names = model.names

    print("-----------------\nDETECTION:")
    farthest_points = []

    for box in boxes:
        cls = int(box.cls[0])
        farthest_points.append((calculate_farthest_point(box), names[cls]))
        
    most_crucial_point = min(farthest_points, key=lambda x: x[0][0]) if len(farthest_points) > 0 else None

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        org = [x1, y1]
        VIP_point = most_crucial_point[0] == calculate_farthest_point(box)

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 255) if not VIP_point else (0, 0, 255), 3)
        cv2.putText(output_frame, names[cls], org, font, fontScale, color, thickness)
        cv2.drawMarker(output_frame, calculate_farthest_point(box), (0, 255, 0) if not VIP_point else (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        print(f"{names[cls]} at {calculate_farthest_point(box)}")
    
    print("--- Prioritizing --- : ", most_crucial_point)
    cv2.imshow(window_name, output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
