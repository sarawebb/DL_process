from DL_process_functions import *
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import json


with open('text_mined_labels_171_and_split.json', 'r') as f:
    data = json.load(f)

# Load the DL_info.csv file
dl_info = pd.read_csv('DL_info.csv')

# Extract the lesion indices from the JSON data
train_lesion_idxs = data['train_lesion_idxs']
print(len(train_lesion_idxs))
val_lesion_idxs = data['val_lesion_idxs']
print(len(val_lesion_idxs))
test_lesion_idxs = data['test_lesion_idxs']
print(len(test_lesion_idxs))

# Map lesion indices to tumor types
train_tumor_types = map_lesion_to_tumor_type(train_lesion_idxs, dl_info)

'''val_tumor_types = map_lesion_to_tumor_type(val_lesion_idxs, dl_info)
test_tumor_types = map_lesion_to_tumor_type(test_lesion_idxs, dl_info)

# Create DataFrames for each set
train_df = pd.DataFrame({
    'lesion_idx': train_lesion_idxs,
    'tumor_type': train_tumor_types
})

val_df = pd.DataFrame({
    'lesion_idx': val_lesion_idxs,
    'tumor_type': val_tumor_types
})

test_df = pd.DataFrame({
    'lesion_idx': test_lesion_idxs,
    'tumor_type': test_tumor_types
})

# Save DataFrames to CSV files
train_df.to_csv('train_tumor_types.csv', index=False)
val_df.to_csv('val_tumor_types.csv', index=False)
test_df.to_csv('test_tumor_types.csv', index=False)

# Save the updated data back to a JSON file if needed
# with open('updated_text_mined_labels.json', 'w') as f:
#     json.dump(data, f)

print("Tumor types have been assigned to the lesion indices.")
'''