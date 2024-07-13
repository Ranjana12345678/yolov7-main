import cv2
import numpy as np
from multiprocessing import Pool

def process_frame(frame):
    # Draw a circle in the middle of the frame
    height, width, _ = frame.shape
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.circle(frame, center, radius, color, thickness)
    return frame

def process_video_frame(frame_path):
    frame = cv2.imread(frame_path)
    processed_frame = processed_frame.append(process_frame(frame))
    return processed_frame

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    temp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        temp += 1
    cap.release()
    return frames, temp

def main():
    video_path = "new_footage.mp4"
    frames, temp = extract_frames(video_path)

    # Parallel processing using multiprocessing Pool
    with Pool() as pool:
        processed_frames = pool.map(process_video_frame, frames)

    # Combine processed frames into a video
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in processed_frames:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    main()
