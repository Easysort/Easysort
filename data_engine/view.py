import cv2
import os

# Set the path to the video file
path_to_data = '/Users/lucasvilsen/Desktop/EasySort/data'
files = os.listdir(path_to_data)
print(os.listdir(path_to_data))
print("Make sure to pick your index")
index = 0
file_to_open = os.path.join(path_to_data, files[index])
print(f"Opening {file_to_open}")

# Open the video file
cap = cv2.VideoCapture(file_to_open)

if (cap.isOpened()== False): 
    print("Error opening video file")
  
# Read until video is completed 
while (cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read()
    print(frame)
    cv2.imshow("frame", frame)
    print(ret)
    if ret == True: 
    # Display the resulting frame 
        cv2.imshow('Frame', frame) 
          
    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
  
# Break the loop 
    else: 
        break
  
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 