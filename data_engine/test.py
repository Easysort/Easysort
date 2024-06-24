import cv2
import os

# Set up the video capture device (default is 0, which is usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Create a directory to save the frames
frame_dir = 'frames'
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    cv2.imwrite(os.path.join(frame_dir, f'frame_{frame_num:06d}.jpg'), frame)
    frame_num += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()