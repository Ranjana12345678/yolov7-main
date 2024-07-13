import os
import subprocess
import imageio
from multiprocessing import Pool
import cv2

def process_frame(frame):
    # Process individual frame using detect.py
    command = f"python detect.py --weights yolov7.pt --save-txt --conf 0.25 --img-size 640 --source {frame}"
    subprocess.run(command, shell=True)

def main():
    video_path = "new_footage.mp4"

    output_folder = video_path.split(".")[0]
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    # Replace 'video.mp4' with your video file path
    # video_path = 'video.mp4'

    # Read the first frame as an image using imageio
    # image = imageio.imread(video_path, index=0)

    # frames = imageio.mimread(video_path)

    # Open the video capture object
    cap = cv2.VideoCapture(video_path)
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video!")
        exit()

    # List to store all frames
    frames = []

    # Read frames until the end of the video
    while True:
    # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame capture was successful (end of video or error)
        if not ret:
            break

        filename = f"{new_folder}/frame_{frame_count:05d}.jpg"  # 5 digit padding

        # Save the frame as an image
        cv2.imwrite(filename, frame)

        frame_count += 1

        # Append the frame to the list
        frames.append(frame)

    # Now you have all frames in the 'frames' list for further processing

    # Close the video capture object
    cap.release()

    # Iterate through each frame
    with Pool() as pool:
        pool.map(process_frame, frames)

if __name__ == "__main__":
    main()





