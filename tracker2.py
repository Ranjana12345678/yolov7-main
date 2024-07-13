import cv2
import csv
import os

def map_function(csv_file_path, video_path):
    """
    Maps each row in the CSV file to process video frames and draw dots on coordinate points.
    
    Parameters:
    csv_file_path (str): Path to the CSV file.
    video_path (str): Path to the input video file.
    
    Returns:
    list: List of dictionaries, each containing frame number as key and processed frame as value.
    """
    cap = cv2.VideoCapture(video_path)
    mapped_data = []
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            object_id = float(row[0])
            frames = eval(row[1])
            coords = eval(row[2])
            frame_data = {}
            for frame, coord_list in zip(frames, coords):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
                ret, img = cap.read()
                if not ret:
                    continue
                
                for point in coord_list:
                    if not isinstance(point, list) or len(point) != 4:
                        continue  # Skip invalid points
                    
                    center = tuple(point)
                    center = ((center[0]+center[2])/2, (center[1]+center[3])/2)
                    color = (0, 255, 0)  # Green color
                    thickness = -1  # Filled circle
                    radius = 5
                    img = cv2.circle(img, center, radius, color, thickness)
                
                frame_data[frame] = img
                if (frame==2):
                    cv2.imshow(img)
                    cv2.waitKey(0)
            
            mapped_data.append(frame_data)

    cap.release()
    return mapped_data

def reduce_function(mapped_data, output_path, video_width, video_height):
    """
    Reduces the mapped data into a single video file.
    
    Parameters:
    mapped_data (list): List of dictionaries from the map function.
    output_path (str): Path to the output video file.
    video_width (int): Width of the video frame.
    video_height (int): Height of the video frame.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (video_width, video_height))

    
    if not isinstance(mapped_data, list):
        raise TypeError("mapped_data should be a list")
    
    if not all(isinstance(data, dict) for data in mapped_data):
        raise TypeError("Each element in mapped_data should be a dictionary")
    
    all_frames = {}
    for data in mapped_data:
        all_frames.update(data)
    
    
    for frame_number in sorted(all_frames.keys()):
        out.write(all_frames[frame_number])

    out.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to run the map-reduce process and draw dots on the video frames.
    """
    csv_file_path = 'D:/yolov7-main/runs/detect/exp9/output.csv'
    video_file_path = 'D:/yolov7-main/runs/detect/exp9/short_footage.mp4'

    # creating a folder name where all the csv of objects will be saved
    last_slash_index = csv_file_path.rfind("/")
    folder = csv_file_path[:last_slash_index]
    output_video_path = os.path.join(folder, "tracked_video.mp4")

    # Get video dimensions
    cap = cv2.VideoCapture(video_file_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Map function
    mapped_data = map_function(csv_file_path, video_file_path)

    # Reduce function
    reduce_function(mapped_data, output_video_path, video_width, video_height)

if __name__ == "__main__":
    main()
