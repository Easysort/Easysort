import cv2
import os

def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")

def video_to_frames(video_path, output_folder):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    # Create the output directory if it does not exist
    create_folder(output_folder)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return

    frame_count = 0
    while True:
        # Read a frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'")

# Example usage
video_path = 'data/video1.mp4'
output_folder = video_path.split(".")[0]
video_to_frames(video_path, output_folder)
