import os
import subprocess
from multiprocessing import Pool

def process_frame(frame):
    # Process individual frame using detect.py
    command = f"python detect.py --weights yolov7.pt --save-txt --conf 0.25 --img-size 640 --source {frame}"
    subprocess.run(command, shell=True)

def main():
    # Path to the video file
    video_path = "new_footage.mp4"


    
    # Create a list of frames from the video
    frames = []
    with open('frames.txt', 'w') as f:
        subprocess.run(f'ffmpeg -i {video_path} -vf "select=not(mod(n\,10))" -vsync vfr frame_%03d.png', shell=True, stdout=f)
    with open('frames.txt', 'r') as f:
        frames = [line.strip() for line in f.readlines()]
    
    # Process frames in parallel
    with Pool() as pool:
        pool.map(process_frame, frames)

if __name__ == "__main__":
    main()