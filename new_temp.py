import csv
import math
import pandas as pd
import ast

tot_dynamic = 0
tot_static = 0
df = pd.read_csv('D:/yolov7-main/runs/detect/exp11/output.csv')

for row in df.itertuples(index=False):
    object_id = row[0]
    frames = eval(row[1])
    coordinates = row[2]
    coordinates = ast.literal_eval(coordinates)

    f_pointx = (coordinates[0][0] + coordinates[0][2])/2
    f_pointy = (coordinates[0][1] + coordinates[0][3])/2

    f1 = f_pointx
    f2 = f_pointy

    # Draw lines connecting the coordinates
    for i in range(len(coordinates) - 1):
        s_pointx = (coordinates[i+1][0] + coordinates[i+1][2])/2
        s_pointy = (coordinates[i+1][1] + coordinates[i+1][3])/2
        
        f_pointx = s_pointx
        f_pointy = s_pointy

    d = abs(math.sqrt(((f1 - f2) ** 2) + ((f_pointx - f_pointy) ** 2)))

    if (d > 100):
        status = "dynamic"
        tot_dynamic += 1
    else:
        status = "static"
        tot_static += 1

    print (object_id, "-", d, " : ", status)

    
print ("------------------------------------------------------------------------------")
total_objects = tot_static + tot_dynamic
if (tot_dynamic > tot_static):
    print ("The crowd is DYNAMIC")
else:
    print ("The crowd is STATIC")