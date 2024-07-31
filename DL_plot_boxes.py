from DL_process_functions import *
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import pandas as pd


input_dir_DL_info_data = sys.argv[1]
norm_image_path = sys.argv[2]
new_path_size_boxes = sys.argv[3]

dl_info_data, file_names = read_dl_info_data(input_dir_DL_info_data, norm_image_path)
draw_bounding_boxes(False, new_path_size_boxes, dl_info_data, file_names, norm_image_path)
