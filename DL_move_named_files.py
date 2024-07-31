from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#testing 
input_dir = sys.argv[1]
output_dir = sys.argv[2]

#subdirs = get_subdirectory_paths(input_dir)
#print(subdirs)

move_folders_based_on_filename_length(input_dir, output_dir, min_length=8)






