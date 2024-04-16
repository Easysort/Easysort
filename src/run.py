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

model = YOLO("/Users/lucasvilsen/Desktop/EasySort/vision/runs/train/weights/best.pt")

# Set the device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = model.to(device)

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)
window_name = 'DETR Object Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # Adjust the size as needed

def calculate_farthest_point(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return (x1+x2)//2, (y1+y2)//2

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blank frame to draw the bounding boxes on
    output_frame = frame.copy()

    # Perform object detection on the frame
    results = model(frame, stream=True)

    # Draw bounding boxes on the output frame
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

    # Display the resulting frame
    cv2.imshow(window_name, output_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
