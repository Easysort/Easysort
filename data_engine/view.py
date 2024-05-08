import cv2
import os

# Set the path to the video file
path_to_data = '/Users/lucasvilsen/Desktop/EasySort/data'
files = os.listdir(path_to_data)
print(os.listdir(path_to_data))
print("Make sure to pick your index")
index = 1

# Open the video file
cap = cv2.VideoCapture(os.path.join(path_to_data, files[index]))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Display the frame
    cv2.imshow("Video", frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Check for the 'q' key to exit
    if key == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()