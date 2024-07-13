import pandas as pd
from PIL import Image, ImageDraw
import ast
import os
import math
import csv


def map_function(row):

    object_id = row[0]
    frames = eval(row[1])
    coordinates = row[2]
    coordinates = ast.literal_eval(coordinates)
    # coordinates = list(coordinates)
    # print (len(coordinates))
    
    # Create a blank image (size can be adjusted)
    img = Image.new('RGB', (1920, 1080), 'white')
    draw = ImageDraw.Draw(img)
    
    f_pointx = (coordinates[0][0] + coordinates[0][2])/2
    f_pointy = (coordinates[0][1] + coordinates[0][3])/2

    f1 = f_pointx
    f2 = f_pointy

    # Draw lines connecting the coordinates
    for i in range(len(coordinates) - 1):
        s_pointx = (coordinates[i+1][0] + coordinates[i+1][2])/2
        s_pointy = (coordinates[i+1][1] + coordinates[i+1][3])/2
        draw.line((f_pointx, f_pointy, s_pointx, s_pointy), fill='black', width=20)
        f_pointx = s_pointx
        f_pointy = s_pointy

    d = math.sqrt(((f1 - f2) ** 2) + ((f_pointx - f_pointy) ** 2))

    if (d > 150):
        status = "dynamic"
        # tot_dynamic += 1
    else:
        status = "static"
        # tot_static += 1

    print (object_id, "-", d, " : ", status)
    data_row = [object_id, d, status]


    folder_path = 'D:/yolov7-main/runs/detect/exp11/output_images'
    os.makedirs(folder_path, exist_ok=True)  # This will create the folder if it doesn't exist

    # Step 6: Save the image to the folder
    file_name = str(object_id) + '.png'
    image_path = os.path.join(folder_path, file_name)
    img.save(image_path)
    return object_id, img, data_row

# def reduce_function(mapped_data):
#     # Create a final blank image to combine all the lines
#     final_img = Image.new('RGB', (1920, 1080), 'white')
#     final_draw = ImageDraw.Draw(final_img)
    
#     for object_id, img in mapped_data:
#         final_img = Image.alpha_composite(final_img.convert('RGBA'), img.convert('RGBA'))
    
#     return final_img

if __name__ == "__main__":

    # Read the CSV file
    csv_file_path = 'D:/yolov7-main/runs/detect/exp11/output.csv'
    df = pd.read_csv(csv_file_path)
    # Display the first few rows to understand the structure
    # print(df.head())


    # creating a folder name where the new csv will be saved
    # last_slash_index = csv_file_path.rfind("/")
    # folder = csv_file_path[:last_slash_index]
    # output_csv_path = os.path.join(folder, "final_output.csv")


    # df1=pd.DataFrame()

    # object_id_li = []
    # distance_li = []


    # df1["Object ID"] = object_id_li
    # df1["Distance"] = distance_li

    output_csv_path = "D:/yolov7-main/runs/detect/exp11/final_output.csv"

    

    dynamic_count = 0
    static_count = 0

    # Apply the map function to each row
    mapped_data = [map_function(row) for row in df.itertuples(index=False)]

    header = ['Object ID', 'Distance', 'Status']
    # Open a file in write mode
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)

        for object_id, img, data_row in mapped_data:
            
            writer.writerow(data_row)

    
    with open(output_csv_path, 'r') as file:
        reader = csv.reader(file)
        
        # Read all rows
        rows = list(reader)
        
        # Check if there are at least 3 rows
        if len(rows) >= 3:
            # Get the third row (index 2)
            third_row = rows[2]
            
            # Count occurrences of "dynamic" and "static"
            dynamic_count = third_row.count('dynamic')
            static_count = third_row.count('static')

    # # Apply the reduce function
    # final_image = reduce_function(mapped_data)
    # # Save or display the final image
    # final_image.show()
    # final_image.save('combined_image.png')

    print ("------------------------------------------------------------------------------")
    total_objects = dynamic_count + static_count

    if (dynamic_count > static_count):
        print ("The crowd is DYNAMIC")
    else:
        print ("The crowd is STATIC")
