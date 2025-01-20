import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = 'path_to_your_downloaded_data/PPG_FieldStudy/'
subject_ids = range(1, 16)  # 15 subjects

# Initialize lists to store data
ecg_data = []
ppg_data = []
acc_data = []
labels = []

# Iterate over each subject's data
for subject_id in subject_ids:
    # Load ECG data
    ecg_file = f'{data_path}S{subject_id:02d}/ECG.mat'
    ecg = scipy.io.loadmat(ecg_file)['ECG']
    ecg_data.append(ecg)
    
    # Load PPG data
    ppg_file = f'{data_path}S{subject_id:02d}/PPG.mat'
    ppg = scipy.io.loadmat(ppg_file)['PPG']
    ppg_data.append(ppg)
    
    # Load Accelerometer data
    acc_file = f'{data_path}S{subject_id:02d}/ACC.mat'
    acc = scipy.io.loadmat(acc_file)['ACC']
    acc_data.append(acc)
    
    # Load labels (e.g., activity types)
    label_file = f'{data_path}S{subject_id:02d}/Label.mat'
    label = scipy.io.loadmat(label_file)['Label']
    labels.append(label)

# Convert lists to numpy arrays
ecg_data = np.concatenate(ecg_data, axis=0)
ppg_data = np.concatenate(ppg_data, axis=0)
acc_data = np.concatenate(acc_data, axis=0)
labels = np.concatenate(labels, axis=0)

# Standardize the data
scaler = StandardScaler()
ecg_data = scaler.fit_transform(ecg_data)
ppg_data = scaler.fit_transform(ppg_data)
acc_data = scaler.fit_transform(acc_data)

# Combine data into a single DataFrame
df = pd.DataFrame({
    'ECG': list(ecg_data),
    'PPG': list(ppg_data),
    'ACC': list(acc_data),
    'Label': labels.flatten()
})

# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)


''' Citation
Reiss, A., Indlekofer, I., & Schmidt, P. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53890.
'''