from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import csv
#testing 
input_dir_DL_info_data = sys.argv[1]
norm_image_path = sys.argv[2]
new_path_size_boxes = sys.argv[3]

dl_info_data, file_names = read_dl_info_data(input_dir_DL_info_data, norm_image_path)

df_names_list = dl_info_data['File_name'].tolist()
common_items = list(set(file_names) & set(df_names_list))
print(len(common_items))

csv_filename = 'df_names_list.csv'
with open(csv_filename, 'w', newline='') as file:
    for name in df_names_list:
        file.write(name + '\n')
print(f"List of names saved to {csv_filename}")

csv_filename = 'png_names_list.csv'
with open(csv_filename, 'w', newline='') as file:
    for name in file_names:
        file.write(name + '\n')
print(f"List of names saved to {csv_filename}")

#draw_bounding_boxes(False, new_path_size_boxes, dl_info_data, file_names, norm_image_path)