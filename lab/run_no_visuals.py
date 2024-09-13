# This script runs the best performing model
# It continuously evaluates the camera
# When an object is found
# it returns the type and midpoint.

# TODO:
# - Be able to give arduino position and speed of object to catch.
# - Be able to listen to incoming call from arduino when ready again.
# - Be able to store the type sorted and how many in db

import cv2
import torch
from ultralytics import YOLO
import time

last_time = time.time()

def has_second_passed():
    global last_time
    current_time = time.time()
    if current_time - last_time >= 1:
        last_time = current_time
        return True
    return False

model = YOLO("/Users/lucasvilsen/Desktop/EasySort/runs/train4/weights/best.pt", verbose = False)

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

    results = model(frame, stream=True, verbose=False)
    result = list(results)[0]
    
    boxes = list(result.boxes)
    names = model.names

    points_strings = []
    farthest_points = []

    for box in boxes:
        cls = int(box.cls[0])
        farthest_points.append((calculate_farthest_point(box), names[cls]))
        points_strings.append(f"{names[cls]} at {calculate_farthest_point(box)}")
        
    most_crucial_point = min(farthest_points, key=lambda x: x[0][0]) if len(farthest_points) > 0 else None

    if has_second_passed():
        print("All points: ", *points_strings, sep="\n")
        print("\n--- Prioritizing --- : \n", most_crucial_point, "\n---  ---")


cap.release()
