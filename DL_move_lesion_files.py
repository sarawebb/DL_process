from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import json


csv_file_path = 'df_names_list.csv'

df = pd.read_csv(csv_file_path)
file_list = df.iloc[:, 0].tolist()

# Define the source and destination directories
source_directory = "/fred/oz141/public_datasets/DeepLesion/named_images/"  # Replace with your source directory path
destination_directory = "/fred/oz141/public_datasets/DeepLesion/lesion_images"  # Replace with your destination directory path

# Call the function to move the files
move_files(file_list, source_directory, destination_directory)

print("File moving process completed.")