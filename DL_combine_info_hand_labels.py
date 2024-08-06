import json
import pandas as pd

# Load the hand-labeled test set
with open('hand_labeled_test_set.json', 'r') as f:
    hand_labeled_data = json.load(f)

# Load the DL_info.csv file
dl_info = pd.read_csv('DL_info.csv')

# Create a list to store the combined data
combined_data = []

# Iterate through the hand-labeled data
for item in hand_labeled_data:
    #file_name = item['file_name']
    lesion_idx = item['lesion_idx']
    print(lesion_idx)
    # Find the corresponding row in DL_info.csv
    dl_info_row = dl_info.iloc[lesion_idx]


    print(dl_info_row['File_name'])
    # Combine the data
    combined_item = {
        #'File_name': file_name,
        'File_name': dl_info_row['File_name'],
        'Lesion_idx': lesion_idx,
        'Hand_labeled_text': item['text'],
        'hand_label_expanded_terms': item['expanded_terms'],
        'Patient_index': dl_info_row['Patient_index'],
        'Study_index': dl_info_row['Study_index'],
        'Series_ID': dl_info_row['Series_ID'],
        'Key_slice_index': dl_info_row['Key_slice_index'],
        'Slice_range': dl_info_row['Slice_range'],
        'Bounding_boxes': dl_info_row['Bounding_boxes'],
        'Measurement_coordinates': dl_info_row['Measurement_coordinates'],
        'Coarse_lesion_type': dl_info_row['Coarse_lesion_type'],
        'Image_size': dl_info_row['Image_size'],
        'Patient_gender': dl_info_row['Patient_gender'],
        'Patient_age': dl_info_row['Patient_age'],
        'Train_Val_Test': dl_info_row['Train_Val_Test']
    }
    
    combined_data.append(combined_item)

# Create a DataFrame from the combined data
combined_df = pd.DataFrame(combined_data)

# Save the combined data to a new CSV file
combined_df.to_csv('combined_hand_labeled_and_dl_info.csv', index=False)

print("Combined data saved to 'combined_hand_labeled_and_dl_info.csv'")