import cv2
import os
import subprocess
from multiprocessing import Pool

def process_frame(frame):
    command = f"python detect.py --weights yolov7.pt --save_txt --conf 0.25 --img-size 640 --source {frame}"
    subprocess.run(command, shell=True)

# def process_frame_wrapper(frame):
#     # Wrapper function to pass to map
#     process_frame(frame)

def main():
    # Path to the video file
    video_path = "new_footage.mp4"

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Divide frames into chunks
    chunk_size = 10  # You can adjust this according to your needs
    frame_chunks = [list(range(i, min(i + chunk_size, total_frames))) for i in range(0, total_frames, chunk_size)]
    
    # Process frames in parallel
    with Pool() as pool:
        for chunk in frame_chunks:
            frames = []
            for frame_number in chunk:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            # Process frames in parallel
            pool.map(process_frame, frames)
    
    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    main()
