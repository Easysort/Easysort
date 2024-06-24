import cv2
import os

# Set up the frame directory
frame_dir = 'frames'
frame_files = sorted(os.listdir(frame_dir))

# Display the video
for frame_file in frame_files:
    frame_path = os.path.join(frame_dir, frame_file)
    frame = cv2.imread(frame_path)
    cv2.imshow('Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()