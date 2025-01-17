import numpy as np
import pandas as pd
import cv2
import multiprocessing
import os


def parse_csv(file_path):
    
    #Parses the CSV file and converts the frame_numbers and coordinates columns.
    
    df = pd.read_csv(file_path)
    df['FrameNo'] = df['FrameNo'].apply(eval)
    df['Coordinates'] = df['Coordinates'].apply(eval)
    return df

import cv2

def map_function(df_chunk, video_path):
    """
    Maps each row in the DataFrame to process video frames and draw dots on coordinate points.
    
    Parameters:
    df_chunk (DataFrame): Chunk of DataFrame to process.
    video_path (str): Path to the input video file.
    
    Returns:
    dict: Dictionary with frame number as key and processed frame as value.
    """
    cap = cv2.VideoCapture(video_path)
    mapped_data = {}
    
    for index, row in df_chunk.iterrows():
        object_id = row['ObjectID']
        frames = row['FrameNo']
        coords = row['Coordinates']
        
        for frame, coord_list in zip(frames, coords):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ret, img = cap.read()
            if not ret:
                continue

            # Ensure coord_list is actually a list of points
            if not isinstance(coord_list, list):  # Change 1
                continue

            for point in coord_list:
                if not isinstance(point, list) or len(point) != 4:  # Change 2
                    continue  # Skip invalid points
                
                center = tuple(point)
                center = ((center[0]+center[2])/2, (center[1]+center[3])/2)
                color = (0, 255, 0)  # Green color
                thickness = -1  # Filled circle
                radius = 5
                img = cv2.circle(img, center, radius, color, thickness)
            
            if frame not in mapped_data:
                mapped_data[frame] = img

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

    all_frames = {}
    for data in mapped_data:
        all_frames.update(data)
    
    for frame_number in sorted(all_frames.keys()):
        out.write(all_frames[frame_number])

    out.release()
    cv2.destroyAllWindows()


class Mapper(multiprocessing.Process):
    """
    Mapper class to handle mapping process in a separate process.
    
    Attributes:
    df_chunk (DataFrame): Chunk of DataFrame to process.
    video_path (str): Path to the input video file.
    out_q (Queue): Output queue to put mapped data.
    """
    def __init__(self, df_chunk, video_path, out_q):
        super().__init__()
        self.df_chunk = df_chunk
        self.video_path = video_path
        self.out_q = out_q

    def run(self):
        """
        Runs the map function on the DataFrame chunk and puts the result in the output queue.
        """
        mapped_data = map_function(self.df_chunk, self.video_path)
        self.out_q.put(mapped_data)

class Reducer(multiprocessing.Process):
    """
    Reducer class to handle reducing process in a separate process.
    
    Attributes:
    in_q (Queue): Input queue to get mapped data.
    output_path (str): Path to the output video file.
    video_width (int): Width of the video frame.
    video_height (int): Height of the video frame.
    """
    def __init__(self, in_q, output_path, video_width, video_height):
        super().__init__()
        self.in_q = in_q
        self.output_path = output_path
        self.video_width = video_width
        self.video_height = video_height

    def run(self):
        """
        Runs the reduce function on the collected mapped data and compiles the final video.
        """
        mapped_data = []
        while not self.in_q.empty():
            mapped_data.append(self.in_q.get())
        
        reduce_function(mapped_data, self.output_path, self.video_width, self.video_height)


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
    

    # Parse the CSV file using pandas
    df = parse_csv(csv_file_path)

    # Create queues for mapper and reducer
    map_out_q = multiprocessing.Queue()
    reduce_in_q = multiprocessing.Queue()

    # Get video dimensions
    cap = cv2.VideoCapture(video_file_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Split dataframe into chunks for mappers
    num_mappers = 4
    df_chunks = np.array_split(df, num_mappers)
    mappers = [Mapper(chunk, video_file_path, map_out_q) for chunk in df_chunks]

    # Start mapper processes
    for mapper in mappers:
        mapper.start()

    # Wait for all mappers to finish
    for mapper in mappers:
        mapper.join()

    # Transfer mapped data to reducer queue
    while not map_out_q.empty():
        reduce_in_q.put(map_out_q.get())

    # Start reducer process
    reducer = Reducer(reduce_in_q, output_video_path, video_width, video_height)
    reducer.start()
    reducer.join()

if __name__ == "__main__":
    main()
