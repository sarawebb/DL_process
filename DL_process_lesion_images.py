from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#testing 
input_dir = sys.argv[1]
txt_file = sys.argv[2]
output_dir = sys.argv[3]



move_matching_files(input_dir, txt_file, output_dir)