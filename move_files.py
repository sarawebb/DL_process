import os
from DL_process_functions import *
import random
import shutil

source_directory = "/fred/oz141/public_datasets/DeepLesion/named_images"
destination_directory = "/fred/oz141/public_datasets/DeepLesion/no_lesion_images"
number_of_files_to_move = 30633

move_random_files(source_directory, destination_directory, number_of_files_to_move)

