
import numpy as np
import os
import cv2
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd

def rename_and_move_images(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Extract directory names from the path and use them to rename the images
            directory_names = root.replace(folder_path, '').strip(os.sep)
            new_filename = directory_names.replace(os.sep, '_') + '_' + file
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(output_folder, new_filename)
            with open(old_file_path, 'rb') as f_read:
                with open(new_file_path, 'wb') as f_write:
                    f_write.write(f_read.read())

def load_and_normalize_image(image_path):
    # Load the image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Normalize the image
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return normalized_image

def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def load_and_normalize_image_minmax(image_path, method='minmax'):
    # Load the image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Normalize the image based on the specified method
    if method == 'minmax':
        normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    elif method == 'zscore':
        mean, std_dev = cv2.meanStdDev(image)
        normalized_image = (image - mean) / std_dev
        normalized_image = normalized_image.astype(np.float32)
    else:
        raise ValueError("Unsupported normalization method. Use 'minmax' or 'zscore'.")
    
    return normalized_image

def save_normalized_image(image, output_folder, filename):
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = output_folder + filename
    print(output_path)
    cv2.imwrite(output_path, (image * 255).astype(np.uint8))

def draw_bounding_boxes(draw_line, new_path, dl_info_data, file_names, directory_path):
    for name in dl_info_data['File_name']:
        if name in file_names: 
            print(name)
            image_path = os.path.join(directory_path, name)
            image = cv2.imread(image_path)
            normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            bounding_boxes = dl_info_data[dl_info_data['File_name'] == name]['Bounding_boxes'].values
            box = list(map(float, bounding_boxes[0].split(', ')))
            print(box[0], box[1],box[2],box[3])
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            color = (0, 0, 255) #red box 
            #color = (255, 0, 0) #blue box 
            #color = (0, 255, 0) #green box

            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            if draw_line == True:
                cv2.line(image, start_point, end_point, color, 1)
            else: 
                pass
            output_image_path = os.path.join(new_path, f"bounding_boxes_{name}")
            cv2.imwrite(output_image_path, image)

        else: 
            pass

def read_dl_info_data(input_dir_DL_info_data, norm_image_path):
    dl_info_data = pd.read_csv(input_dir_DL_info_data)
    file_names = [file for file in os.listdir(norm_image_path) if file.endswith('.png')]
    return dl_info_data, file_names


def generate_sh_files(paths):
    script_template = """#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH --mem=100 # memory in Mb

#SBATCH --output=rename.out
#SBATCH --error=rename.err

#SBATCH --partition=skylake
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module load mamba

mamba activate py3.9

python /fred/oz141/public_datasets/scripts/process_DL/DL_rename_images_infolder.py {path}
"""
    for i, path in enumerate(paths):
        script_content = script_template.format(path=path)
        with open(f"script_{i}.sh", "w") as file:
            file.write(script_content)
        print(f"Generated script_{i}.sh for path: {path}")

# Example usage:
# paths = ['/path/to/dataset1', '/path/to/dataset2']
# generate_sh_files(paths)

import os
import subprocess

def run_sbatch_scripts(directory):
    for file in os.listdir(directory):
        if file.endswith(".sh"):
            command = f"sbatch {file}"
            subprocess.run(command, shell=True, check=True)



# Function to map lesion indices to tumor types
def map_lesion_to_tumor_type(lesion_idxs, dl_info):
    print('In function')
    tumor_types = []
    for idx in lesion_idxs:
        print('------------')
        print(idx)
        print('------------')
        mask = dl_info.loc[idx]
        #print(len(mask))
        
        #tumor_type = dl_info.loc[dl_info['lesion_idx'] == idx, 'tumor_type'].values
        #if len(tumor_type) > 0:
         #   tumor_types.append(tumor_type[0])
        #else:
         #   tumor_types.append(None)  # or some default value
    return tumor_types

def move_files(file_list, source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Walk through the source directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file in file_list:
                print(file)
                # Construct the full path of the source file
                source_path = os.path.join(root, file)
                print(source_path)
                # Construct the full path of the destination file
                destination_path = os.path.join(destination_dir, file)
                # Move the file
                os.system('mv ' + source_path + ' '+ destination_path)
                print(f"Moved {file} to {destination_path}")

