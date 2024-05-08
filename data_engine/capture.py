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

# Initialize variables
recording = False
out = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Create a copy of the frame for display
    display_frame = frame.copy()

    # Draw a rectangle around the recording button
    if recording:
        display_frame = cv2.rectangle(display_frame, (10, 10), (50, 30), (0, 0, 255), 2)  # Red if recording
    else:
        display_frame = cv2.rectangle(display_frame, (10, 10), (50, 30), (128, 128, 128), 2)  # Grey if not recording

    # Display the frame
    cv2.imshow("Camera", display_frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Check for the 'q' key to exit
    if key == ord('q'):
        break

    # Check for the 'r' key to start recording
    if key == ord('r'):
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(os.path.join(data_folder, f'recorded_video{int(time.time())}.avi'), fourcc, 20.0, (640, 480))
            print("Recording started. Press 's' to stop recording.")
            recording = True
        else:
            print("Already recording")

    # Check for the 's' key to stop recording
    if key == ord('s') and recording:
        out.release()
        print("Recording stopped.")
        recording = False
        out = None

    # Write the frame to the video file if recording
    if recording:
        out.write(frame)

    # Add a delay to ensure the window is updated
    cv2.waitKey(1)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()