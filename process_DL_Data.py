
import numpy as np
#import nibabel as nib
import os
import cv2
import csv
import glob
import sys

import matplotlib.pyplot as plt

def rename_and_move_images(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if file exists in directory in loop
            directory_names = root.replace(folder_path, '').strip(os.sep)
            new_filename = directory_names.replace(os.sep, '_') + '_' + file
            if os.path.isfile(os.path.join(output_folder, new_filename)):
                print(f"{new_filename} exists in the directory.")
            else:
                print(f"{new_filename} does not exist in the directory.")

            # Extract directory names from the path and use them to rename the images
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


def process_images(output_path):
    image_files = glob.glob(os.path.join(output_path, '*.png'))  # Assuming images are in PNG format

    for image_file in image_files:
        normalized_image = load_and_normalize_image(image_file)
        #print(normalized_image)
        output_filename = os.path.basename(image_file)
        output_path_new = os.path.join(output_path, output_filename)
        cv2.imwrite(output_path_new, (normalized_image * 255).astype(np.uint8))

def process_subdirectories(image_path, output_path):
    subdirectories = [x[0] for x in os.walk(image_path)]
    print(subdirectories)
    for subdir in subdirectories:
        print(subd)
        process_images(output_path)

# Input directories from command line
input_dir_1 = sys.argv[1]
input_dir_2 = sys.argv[2]

process_subdirectories(input_dir_1, input_dir_2)
