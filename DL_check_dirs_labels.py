from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#testing 
input_dir = sys.argv[1]
move_dir = sys.argv[2]

move_folders_based_on_filename_length(input_dir, move_dir, min_length=7)

