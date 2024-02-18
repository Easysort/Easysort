import cv2
import torch
from ultralytics import YOLO

model = YOLO("/Users/lucasvilsen/Desktop/ReUse/vision/runs/train/weights/best.pt")

# Set the device to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
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
        result.show()  # display to screen

    # Display the resulting frame
    cv2.imshow('DETR Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
