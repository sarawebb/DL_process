import numpy 
from DL_process_functions import *
from load_ct_img import *
import pandas as pd

# Load the CSV file
csv_file_path = 'DL_info.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file_path)

# Get the first row
first_row = df.iloc[0]

print("First row of the CSV file:")
print(first_row)
image_path = '/fred/oz141/public_datasets/DeepLesion/lesion_images/'

imname = image_path+first_row['File_name']
print(imname)
im, im_scale = load_prep_img(imname, first_row['Key_slice_index'], scale=1.0, max_im_size=512)
output_filename = imname+'_processed.png'
cv2.imwrite(output_filename, im)
print('------------')
print(output_filename)