
import numpy as np
from astropy.stats import sigma_clip
import os
import cv2
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

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
    image.astype(np.int32) - 32768
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Normalize the image
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return normalized_image

def load_image(image_path):
    # Load the image using cv2
    image = cv2.imread(image_path)
    image.astype(np.int32) - 32768
    return image

def load_image_32bit(image_path):
    # Load the image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Convert the image to 32-bit float
    image = image.astype(np.int32)
    return image

def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def load_and_normalize_image_minmax(image_path, method='minmax'):
    # Load the image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image.astype(np.int32) - 32768
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
    cv2.imwrite(output_path, (image * 20535).astype(np.uint16), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #cv2.imwrite(output_path, (image * 255).astype(np.uint8))high_contrast_image = (high_contrast_image * 255).astype(np.uint8)
    #cv2.imwrite(output_path, image)

def save_scaled_image(image, output_path, filename, min_val=0, max_val=500):
    """
    Save the image with values scaled between 0-500 using cv2.
    
    Args:
    image (numpy.ndarray): Input image array.
    output_path (str): Path to save the output image.
    min_val (int): Minimum value for scaling (default: 0).
    max_val (int): Maximum value for scaling (default: 500).
    """
    # Ensure the image is within the desired range
    scaled_image = np.clip(image, min_val, max_val)
    
    # Scale the image to 0-255 for 8-bit image format
    scaled_image = ((scaled_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Save the image
    output_path = output_path + filename
    cv2.imwrite(output_path, (scaled_image))
    
    print(f"Scaled image saved to {output_path}")

    
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

def adjust_pixel_intensity(im):
    return im.astype(np.int32) - 32768

# Modify the image loading process
def load_image(file_path):
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    return adjust_pixel_intensity(im)
    
def load_and_preprocess_image(file_path):
    # Load the image
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.int32) - 32768
    
    # Perform sigma clipping
    clipped_data = sigma_clip(im, sigma=3, maxiters=5)
    
    # Get the lower and upper bounds of the clipped data
    lower_bound = clipped_data[clipped_data > 0].min() 
    #lower_bound = clipped_data.min() 
    upper_bound = clipped_data.max()
    
    # Normalize the clipped data to [0, 1] range
    normalized_im = (im - lower_bound) / (upper_bound - lower_bound)
    normalized_im = np.clip(normalized_im, 0, 1)
    
    return normalized_im

def load_and_preprocess_image_score(file_path):
    # Load the image
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.int32) - 32768
    
    # Perform sigma clipping
    clipped_data = sigma_clip(im, sigma=3, maxiters=5)
    
    # Calculate mean and standard deviation of the clipped data
    mean = np.mean(clipped_data)
    std = np.std(clipped_data)
    
    # Apply Z-score normalization
    normalized_im = (im - mean) / std
    
    return normalized_im

def load_and_preprocess_image_con(file_path):
    # Load the image
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.int32) - 32768
    
    # Perform sigma clipping
    clipped_data = sigma_clip(im, sigma=3, maxiters=10, stdfunc='std')
    
    # Get the lower and upper bounds of the clipped data
    lower_bound = clipped_data.min()
    upper_bound = clipped_data.max()
    
    # Normalize the clipped data to [0, 1] range
    normalized_im = (im - lower_bound) / (upper_bound - lower_bound)
    normalized_im = np.clip(normalized_im, 0, 1)
    
    # Increase contrast using histogram equalization
    normalized_im = (normalized_im * 255).astype(np.uint8)
    contrast_im = cv2.equalizeHist(normalized_im)
    contrast_im = contrast_im.astype(float) / 255.0
    
    return contrast_im

def load_and_preprocess_image_zscore(file_path):
    # Load the image
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.float32) - 32768
    
    # Create a mask for non-zero values
    non_zero_mask = im != 0
    
    # Perform z-score normalization only on non-zero values
    im_non_zero = im[non_zero_mask]
    mean = np.mean(im_non_zero)
    std = np.std(im_non_zero)
    im_zscore = np.zeros_like(im)
    im_zscore[non_zero_mask] = (im_non_zero - mean) / std
    
    # Clip values to a reasonable range (e.g., [-3, 3]) and rescale to [0, 1]
    im_normalized = np.clip(im_zscore, -3, 3)
    im_normalized = (im_normalized - im_normalized.min()) / (im_normalized.max() - im_normalized.min())
    
    return im_normalized

def plot_pixel_histogram(image_path, bins=256, title='Pixel Value Histogram'):
    """
    Plot a histogram of pixel values for the given image.
    
    Args:
    image_path (str): The path to the input image file.
    bins (int): Number of bins for the histogram.
    title (str): Title for the plot.
    """
    # Load the image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    
    # Flatten the image array
    flat_image = image.flatten()
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flat_image, bins=bins, range=(flat_image.min(), flat_image.max()), density=True, alpha=0.7)
    plt.savefig(image_path+'histogram.png')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.close()
    
    
def load_and_preprocess_image_percentile(file_path, lower_percentile=1, upper_percentile=99):
    # Load the image
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.float32) - 32768
    
    # Create a mask for non-zero values
    non_zero_mask = im != 0
    
    # Perform percentile normalization only on non-zero values
    im_non_zero = im[non_zero_mask]
    lower_bound = np.percentile(im_non_zero, lower_percentile)
    upper_bound = np.percentile(im_non_zero, upper_percentile)
    
    # Clip the image values to the percentile range
    im_clipped = np.clip(im, lower_bound, upper_bound)
    
    # Normalize to [0, 1] range
    im_normalized = (im_clipped - lower_bound) / (upper_bound - lower_bound)
    
    return im_normalized

def load_and_preprocess_image_minmax_nozero(file_path):
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.int32) - 32768
    
    # Create a mask for non-zero values
    non_zero_mask = im != 0
    
    # Find min and max of non-zero values
    min_val = np.min(im[non_zero_mask])
    max_val = np.max(im)
    
    # Apply min-max normalization, keeping original zero values as zero
    normalized_im = np.zeros_like(im, dtype=np.float32)
    normalized_im[non_zero_mask] = (im[non_zero_mask] - min_val) / (max_val - min_val)
    
    # Preserve original zero values
    zero_mask = im == 0
    normalized_im[zero_mask] = 0
    
    return normalized_im


def load_and_preprocess_image_sigma_clip(file_path, sigma=3, iterations=2):
    # Load the image
    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    
    # Subtract 32768 from pixel intensity
    im = im.astype(np.float32) - 32768
    
    # Create a mask for non-zero values
    non_zero_mask = im != 0
    
    # Apply sigma clipping to non-zero values
    clipped_im = im.copy()
    for _ in range(iterations):
        mean = np.mean(clipped_im[non_zero_mask])
        std = np.std(clipped_im[non_zero_mask])
        clip_mask = np.logical_and(clipped_im > mean - sigma * std, 
                                   clipped_im < mean + sigma * std)
        clipped_im[~clip_mask] = mean
    
    # Normalize the clipped image between 0 and 1
    min_val = np.min(clipped_im[non_zero_mask])
    max_val = np.max(clipped_im)
    normalized_im = np.zeros_like(clipped_im)
    normalized_im[non_zero_mask] = (clipped_im[non_zero_mask] - min_val) / (max_val - min_val)
    
    # Increase contrast for non-zero values
    contrast_factor = 1.5
    normalized_im[non_zero_mask] = np.clip(contrast_factor * (normalized_im[non_zero_mask] - 0.5) + 0.5, 0, 1)
    
    return normalized_im
   
def normalise(image):
    # normalise and clip images -1000 to 800
    np_img = image
    
    np_img = np.clip(np_img, -200., 500.).astype(np.float32)
    return np_img

def normalise_q1(image):
    np_img = image
    q1 = np.quantile(np_img, 0.25)
    q2 = np.quantile(np_img, 0.50)
    q3 = np.quantile(np_img, 0.75)
    q4 = np.quantile(np_img, 1)
    np_img = np.clip(np_img, q3, q4).astype(np.float32)
    return np_img

def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret

def save_iamge(image, output_path, filename):
    
    output_path = output_path + filename
    cv2.imwrite(output_path, (image))
    
    print(f"Image saved to {output_path}")

def normalise_hist(image):
    # normalise and clip images -1000 to 800
    np_img = image
    
    # Create histogram before clipping
    hist, bins = np.histogram(np_img.flatten(), bins=100)
    plt.figure(figsize=(10, 5))
    plt.hist(np_img.flatten(), bins=100)
    plt.title('Histogram before clipping')
    plt.xlabel('Pixel values')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig('Test_hist.png')
    
    #np_img = np.clip(np_img, -1000., 300.).astype(np.float32)
    # Increase contrast by adjusting the clipping range
    np_img = np.clip(np_img, -800., 200.).astype(np.float32)
    

    return np_img

def apply_windowing(image, window1, window2):
    """
    Apply intensity windowing to the input image.
    
    Args:
    image (numpy.ndarray): Input image in HU values.
    window_center (float): The center of the window.
    window_width (float): The width of the window.
    
    Returns:
    numpy.ndarray: Windowed image with values scaled to 0-255.
    """
    #window_min = window_center - window_width // 2
    #window_max = window_center + window_width // 2
    try: 
        windowed_image = np.clip(image, window1, window2)
    except: 
         windowed_image = np.clip(image, -800, 500)
    
    # Scale to 0-255 and convert to uint8
    #windowed_image = ((windowed_image - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    
    return windowed_image

def process_and_save_image(image, output_path, filename, window1, window2):
    """
    Load an image, apply windowing, and save the result.
    
    Args:
    image_path (str): Path to the input image.
    output_path (str): Path to save the processed image.
    window_center (float): The center of the window.
    window_width (float): The width of the window.
    """
    # Load the image
    #image = load_image_32bit(image_path)
    
    # Apply windowing
    windowed_image = apply_windowing(image, window1, window2)
    
    # Save the windowed image
     # Save the image
    output_path = output_path + filename
    cv2.imwrite(output_path, (windowed_image))

    #v2.imwrite(output_path, (windowed_image))
    #print(f"Image saved to {output_path}")

# Example usage:
# process_and_save_image('input_image.png', 'output_image.png', 40, 400)


