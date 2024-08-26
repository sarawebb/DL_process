from DL_process_functions import *
import numpy as np
#import nibabel as nib
import matplotlib.pyplot as plt
import sys
import pandas as pd
from tqdm import tqdm


input_path = sys.argv[1]
output_folder = sys.argv[2]

for filename in tqdm(os.listdir(input_path)):
    file_path = os.path.join(input_path, filename)
    if os.path.isfile(file_path):
       
        window1 = -200
        window2 = 500
       
    
        # Check if the file exists
        if os.path.isfile(file_path):
            print(f"The file '{filename}' already exists in the output path.")
        return True
        else:
            print(f"The file '{filename}' does not exist in the output path.")
        
            im_load = load_image(file_path)
            #window_image = apply_windowing(image, window1, window2)
            process_and_save_image(im_load, output_folder, filename,window1, window2)
        #norm_image = normalise(im_load)
        #save_scaled_image(norm_image, output_folder, filename)
    
