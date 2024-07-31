from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#testing 
input_dir = sys.argv[1]
output_file = sys.argv[2]

directories_with_subdirs = check_for_subdirectories_in_directories(input_dir, output_file)
print(directories_with_subdirs)

