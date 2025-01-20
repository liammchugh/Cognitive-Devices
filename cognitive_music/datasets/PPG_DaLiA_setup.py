import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# NOTE: How to download the dataset
# ---------------------------------
# The PPG-DaLiA dataset can be downloaded from the UCI Machine Learning Repository:
# Link: https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA
# After downloading, extract the dataset into a directory (e.g., "path_to_your_downloaded_data/PPG_FieldStudy/")
# Ensure the structure remains intact so that the script can locate the .pkl files.

# Define the data directory
data_dir = r"datasets/ppg+dalia/data/PPG_FieldStudy/"

# Initialize lists to store processed data
ppg_data = []
acc_data = []
labels = []
activities = []
subject_ids_list = []
ages = []
genders = []
heights = []
weights = []
skins = []
sports = []

# Process data for each subject
subject_ids = range(1, 4)  # Subjects S1 to S15
for subject_id in subject_ids:
    # try:
    # "C:\Users\liams\Documents\GitHub\Emotive-Cognition\cognitive_music\datasets\ppg+dalia\data\PPG_FieldStudy\S1\S1.pkl"        # Load the synchronized .pkl file
    subject_path = os.path.join(data_dir, f"S{subject_id}/S{subject_id}.pkl")
    # Load the synchronized .pkl file with the correct encoding
    with open(subject_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")  # Specify the encoding

    
    # Extract relevant data
    ppg = np.array(data["signal"]["wrist"]["BVP"])  # PPG signal (64 Hz)
    acc = np.array(data["signal"]["wrist"]["ACC"])  # Accelerometer signal (32 Hz)
    hr_label = np.array(data["label"])  # Ground truth heart rate
    activity = np.array(data["activity"])  # Activity labels
    
    # Append to lists
    ppg_data.append(ppg)
    acc_data.append(acc)
    labels.append(hr_label)
    activities.append(activity)
    
    # Load subject-specific information from SX_quest.csv
    quest_file = os.path.join(data_dir, f"S{subject_id}/S{subject_id}_quest.csv")

    try:
        # Read the file while removing the '#' character
        with open(quest_file, "r") as f:
            lines = [line.strip().lstrip("#").strip() for line in f.readlines() if line.strip()]  # Remove '#' and empty lines
        
        # Convert to dictionary
        subject_info = {}
        for line in lines:
            key, value = line.split(",", 1)  # Split by the first comma
            subject_info[key.strip()] = value.strip()

        # Extract individual attributes
        subject_id = subject_info.get("SUBJECT_ID", f"S{subject_id}")
        age = int(subject_info.get("AGE", -1))
        gender = subject_info.get("GENDER", "Unknown")
        height = int(subject_info.get("HEIGHT", -1))
        weight = int(subject_info.get("WEIGHT", -1))
        skin = int(subject_info.get("SKIN", -1))
        sport = int(subject_info.get("SPORT", -1))

    except Exception as e:
        print(f"Error reading {quest_file}: {e}")
        continue  # Skip to the next subject if there's an issue

    # Debug output
    print(f"Subject {subject_id} info: {subject_info}")


    # except Exception as e:
    #     print(f"Error processing subject S{subject_id}: {e}")

# Combine data across all subjects
ppg_data = np.concatenate(ppg_data, axis=0)
acc_data = np.concatenate(acc_data, axis=0)
hr_labels = np.concatenate(labels, axis=0)
activities = np.concatenate(activities, axis=0)

# Prepare DataFrame for saving
df = pd.DataFrame({
    "PPG": list(ppg_data),
    "ACC": list(acc_data),
    "HeartRate": hr_labels,
    "Activity": activities,
    "SubjectID": subject_ids_list,
    "Age": ages,
    "Gender": genders,
    "Height": heights,
    "Weight": weights,
    "SkinType": skins,
    "SportLevel": sports,
})

# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save processed data
os.makedirs("PPG_ACC_processed_data", exist_ok=True)
df.to_csv("PPG_ACC_processed_data/data.csv", index=False)
train_df.to_csv("PPG_ACC_processed_data/train_data.csv", index=False)
test_df.to_csv("PPG_ACC_processed_data/test_data.csv", index=False)

print("Processed data saved to train_data.csv and test_data.csv")
