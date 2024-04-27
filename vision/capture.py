import cv2
import os
import time

# Set the path to the "data" folder
data_folder = 'data'

# Check if the "data" folder exists, create it if it doesn't
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Open the camera
cap = cv2.VideoCapture(0)
recording = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Camera', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check for the 'r' key to start recording
    if cv2.waitKey(1) & 0xFF == ord('r'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(data_folder, f'recorded_video{int(time.time())}.avi'), fourcc, 20.0, (640, 480))
        print("Recording started. Press 's' to stop recording.")
        recording = True
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        out.release()
        print("Recording stopped.")
        recording = False
        break

    # Draw a rectangle around the recording button
    if recording:
        cv2.rectangle(frame, (10, 10), (50, 30), (0, 0, 255), 2)
    cv2.imshow('Camera', frame)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()