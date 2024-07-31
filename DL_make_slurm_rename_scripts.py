from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#testing 
input_dir = sys.argv[1]

subdirs = get_subdirectory_paths(input_dir)
print(subdirs)

for i in subdirs: 
    #print(i)
    sub_subdirs = get_subdirectory_paths(i)
    #print(sub_subdirs)
    generate_sh_files(sub_subdirs)
#rename_images(input_dir)









