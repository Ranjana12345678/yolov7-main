import pandas as pd
from functools import reduce
from multiprocessing import Pool
import os

# Function to process each row and write to separate CSV
def process_row(row, folder):
    id_val = row[0]
    col2_values = row[1]
    col3_values = row[2]
    
    # processing needed for each row
    # Create folder
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Write to CSV
    filename = os.path.join(folder, f"{float(id_val)}.csv")  # IDs are floats
    df = pd.DataFrame({'FrameNo': col2_values, 'Coordinates': col3_values})
    df.to_csv(filename, index=False)
    print(f"Processed row with ID: {id_val}")

# Function to read CSV and apply processing to each row
def process_csv(filename, folder):
    df = pd.read_csv(filename)
    
    # Convert columns to appropriate types (FrameNo and Coordinates are the titles given for the columns)
    df['FrameNo'] = df['FrameNo'].apply(eval)
    df['Coordinates'] = df['Coordinates'].apply(eval)
    
    # Convert DataFrame to list of rows
    rows = df.values.tolist()
    
    # Using multiprocessing Pool to parallelize processing
    with Pool() as pool:
        pool.starmap(process_row, [(row, folder) for row in rows])

# Main function to initiate map-reduce process
def main():
    source_csv = 'D:/yolov7-main/runs/detect/exp9/output.csv'  # Location of the CSV file

    # creating a folder name where all the csv of objects will be saved
    last_slash_index = source_csv.rfind("/")
    folder = source_csv[:last_slash_index]
    folder = folder + "/reduced_CSV"


    process_csv(source_csv, folder)

if __name__ == "__main__":
    main()
