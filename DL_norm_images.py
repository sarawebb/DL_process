from DL_process_functions import *
import numpy as np
#import nibabel as nib
import matplotlib.pyplot as plt
import sys
import pandas as pd
from tqdm import tqdm


input_path = sys.argv[1]
output_folder = sys.argv[2]

csv_file_path = '/fred/oz141/public_datasets/scripts/process_DL/DL_info_windows.csv'  # Replace with the actual path to your CSV file
dl_dataframe = pd.read_csv(csv_file_path) 

for filename in tqdm(os.listdir(input_path)):
    file_path = os.path.join(input_path, filename)
    if os.path.isfile(file_path):
        #print(filename)
        frame = dl_dataframe[dl_dataframe['File_name'] == filename]
        try: 
            window1 = float(frame['DICOM_window_1'])
            window2 = float(frame['DICOM_window_2'])
        except: 
            window1 = -200
            window2 = 500

        im_load = load_image(file_path)
        #window_image = apply_windowing(image, window1, window2)
        process_and_save_image(im_load, output_folder, filename,window1, window2)
        #norm_image = normalise(im_load)
        #save_scaled_image(norm_image, output_folder, filename)
    

