import time
import pandas as pd
from PIL import Image, ImageDraw
import ast
import os
import math
import csv
import cv2


# Start timing
start_time = time.time()

def map_function(row, video_path):
    object_id = row[0]
    frames = eval(row[1])
    last_frame = frames[-1]
    
    coordinates = ast.literal_eval(row[2])

    # Acquire the image of the last frame
    last_frame_img = extract_frame(video_path, last_frame-1)

    if last_frame_img is None:
        return None

    # Convert the frame to a PIL image
    last_frame_img_pil = Image.fromarray(cv2.cvtColor(last_frame_img, cv2.COLOR_BGR2RGB))
    last_frame_draw = ImageDraw.Draw(last_frame_img_pil)

    # Create a blank image (size can be adjusted)
    img = Image.new('RGB', (1920, 1080), 'white')
    draw = ImageDraw.Draw(img)

    f_pointx = (coordinates[0][0] + coordinates[0][2]) / 2
    f_pointy = (coordinates[0][1] + coordinates[0][3]) / 2

    f1 = f_pointx
    f2 = f_pointy

    # Draw lines connecting the coordinates
    for i in range(len(coordinates) - 1):
        s_pointx = (coordinates[i + 1][0] + coordinates[i + 1][2]) / 2
        s_pointy = (coordinates[i + 1][1] + coordinates[i + 1][3]) / 2

        draw.line((f_pointx, f_pointy, s_pointx, s_pointy), fill='black', width=5)
        last_frame_draw.line((f_pointx, f_pointy, s_pointx, s_pointy), fill='white', width=5)

        f_pointx = s_pointx
        f_pointy = s_pointy

    d = math.sqrt(((f1 - f_pointx) ** 2) + ((f2 - f_pointy) ** 2))

    if d > 36:
        status = "dynamic"
    else:
        status = "static"

    print(object_id, "-", d, " : ", status)
    data_row = [object_id, d, status]

    folder_path = 'D:/yolov7-main/runs/detect/exp200/output_images'
    os.makedirs(folder_path, exist_ok=True)

    frame_folder_path = 'D:/yolov7-main/runs/detect/exp200/tracked_images'
    os.makedirs(frame_folder_path, exist_ok=True)

    # Save the images to the folder
    file_name = str(object_id) + '.png'

    frame_image_path = os.path.join(frame_folder_path, file_name)
    image_path = os.path.join(folder_path, file_name)

    last_frame_img_pil.save(frame_image_path)  # Save the frame image
    img.save(image_path)  # Save the blank image

    return object_id, img, data_row

def extract_frame(video_path, frame_number):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Set the frame position
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = video_capture.read()

    # Release the video capture object
    video_capture.release()

    if success:
        return frame
    else:
        print(f"Failed to extract frame {frame_number+1}")
        return None

if __name__ == "__main__":
    # Read the CSV file
    csv_file_path = 'D:/yolov7-main/runs/detect/exp200/output.csv'
    df = pd.read_csv(csv_file_path)

    # Path of the video to extract frames from
    video_path = 'D:/yolov7-main/runs/detect/exp200/ucsd11.mp4'

    # Folder where the new CSV will be saved
    last_slash_index = csv_file_path.rfind("/")
    folder = csv_file_path[:last_slash_index]
    output_csv_path = os.path.join(folder, "final_output.csv")

    dynamic_count = 0
    static_count = 0

    # Apply the map function to each row
    mapped_data = [map_function(row, video_path) for row in df.itertuples(index=False)]

    header = ['Object ID', 'Distance', 'Status']
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for result in mapped_data:
            if result:
                object_id, img, data_row = result
                writer.writerow(data_row)

    # Calculate how many dynamic and static objects are present
    with open(output_csv_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if len(rows) > 1:
            for row in rows[1:]:
                if 'dynamic' in row:
                    dynamic_count += 1
                if 'static' in row:
                    static_count += 1

    print("------------------------------------------------------------------------------")
    total_objects = dynamic_count + static_count

    if dynamic_count > static_count:
        print("The crowd is DYNAMIC")
    else:
        print("The crowd is STATIC")




# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time} seconds")