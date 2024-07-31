from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#testing 
input_dir = sys.argv[1]

#subdirs = get_subdirectory_paths(input_dir)
#print(subdirs)

#generate_sh_files(subdirs)
rename_images(input_dir)
