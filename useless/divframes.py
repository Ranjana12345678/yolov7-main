import cv2
import subprocess
import numpy as np
from multiprocessing import Pool

def extract_frames(video_path, frame_rate=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize frame count
    frame_count = 0
    
    # Loop through each frame in the video
    frames = []
    while True:
        # Read the next frame
        ret, frame = cap.read()
        
        # Check if the frame was successfully read
        if not ret:
            break
        
        # Skip frames according to the frame_rate
        if frame_count % frame_rate != 0:
            frame_count += 1
            continue
        
        # Append the frame to the list
        frames.append(frame)
        
        # Increment the frame count
        frame_count += 1
        
        # Break the loop if we have reached the end of the video
        if frame_count >= total_frames:
            break
    
    # Release the video capture object
    cap.release()
    
    # Convert frames list to NumPy array
    frames = np.array(frames)
    
    return frames

def process_frame(frame):
    command = f"python detect.py --weights yolov7.pt --save-txt --conf 0.25 --img-size 640 --source {frame}"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    # Path to the input video file
    video_path = "new_footage.mp4"
    
    # Frame rate at which to extract frames (e.g., 1 frame per second)
    frame_rate = 1
    
    # Extract frames from the video
    frames = extract_frames(video_path, frame_rate)
    
    # Now you have frames as a NumPy array
    # print("Number of frames:", len(frames))

    total = len(frames)

    with Pool() as pool:
        for num in range (0, total):
            pool.map(process_frame, frames[num])
