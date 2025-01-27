import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def prep_training_data():

    """
    # NOTE: How to download the dataset
    # ---------------------------------
    # The PPG-DaLiA dataset can be downloaded from the UCI Machine Learning Repository:
    # Link: https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA
    # After downloading, extract the dataset into a directory (e.g., "path_to_your_downloaded_data/PPG_FieldStudy/")
    # Ensure the structure remains intact so that the script can locate the .pkl files.
    """
    # Define the data directory
    data_dir = r"cognitive_music/datasets/ppg+dalia/data/PPG_FieldStudy/"

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
    subject_ids = range(1, 16)  # Subjects S1 to S15
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
        # print(activity) if subject_id == 1 else None

        # Append to lists
        ppg_data.append(ppg)
        
        # Interpolate data to match the length of the PPG signal
        acc_interp = np.zeros((ppg.shape[0], acc.shape[1]))
        for i in range(acc.shape[1]):
            acc_interp[:, i] = np.interp(
                np.linspace(0, len(acc) - 1, num=len(ppg)),
                np.arange(len(acc)),
                acc[:, i]
            )
        acc = acc_interp
        acc_data.append(acc)

        # Interpolate HR labels
        hr_label_interp = np.zeros(ppg.shape[0])
        hr_label_interp = np.interp(
            np.linspace(0, len(hr_label) - 1, num=len(ppg)),
            np.arange(len(hr_label)),
            hr_label
        )
        labels.append(hr_label_interp)

        # Interpolate activity labels to match the length of the PPG signal
        activities_interp = np.round(np.interp(
            np.linspace(0, len(activity) - 1, num=ppg.shape[0]),  # Target indices
            np.arange(len(activity)),  # Original indices
            activity.flatten()  # Flatten activity array to ensure it's 1D
        )).astype(int)

        activities.append(activities_interp)

        
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

        # For length of current subject ppg data, append subject info to lists
        subject_ids_list.extend([subject_id] * len(ppg))
        ages.extend([age] * len(ppg))
        genders.extend([gender] * len(ppg))
        heights.extend([height] * len(ppg))
        weights.extend([weight] * len(ppg))
        skins.extend([skin] * len(ppg))
        sports.extend([sport] * len(ppg))


        # except Exception as e:
        #     print(f"Error processing subject S{subject_id}: {e}")

    # Combine data across all subjects
    ppg_data = np.concatenate(ppg_data, axis=0)
    acc_data = np.concatenate(acc_data, axis=0)
    hr_labels = np.concatenate(labels, axis=0)
    activities = np.concatenate(activities, axis=0)

    print(f"PPG data shape: {ppg_data.shape}")
    print(f"Accelerometer data shape: {acc_data.shape}")
    print(f"HR labels shape: {hr_labels.shape}")
    print(f"Activity labels shape: {activities.shape}")

    # Dictionary mapping activity IDs to their descriptions
    activity_labels = {
        0: "Transition",  # For transient periods or unlabelled data
        1: "Sitting and Reading",
        2: "Climbing Stairs",
        3: "Playing Table Soccer",
        4: "Cycling Outdoors",
        5: "Driving a Car",
        6: "Lunch Break",
        7: "Walking",
        8: "Working at Desk"
    }

    # Map activity IDs to their descriptions
    activity_descr = [activity_labels[act_id] for act_id in activities]

    # Prepare DataFrame for saving
    df = pd.DataFrame({
        "PPG": list(ppg_data),
        "ACCi": list(acc_data[:, 0]),
        "ACCj": list(acc_data[:, 1]),
        "ACCk": list(acc_data[:, 2]),
        "HeartRate": hr_labels,
        "Activity": activities,
        "ActivityDescr": activity_descr,
        "SubjectID": subject_ids_list,
        "Age": ages,
        "Gender": genders,
        "Height": heights,
        "Weight": weights,
        "SkinType": skins,
        "SportLevel": sports,
    })

    # Split into training and testing sets
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save processed data
    os.makedirs("cognitive_music/datasets/PPG_ACC_processed_data", exist_ok=True)
    df.to_csv("cognitive_music/datasets/PPG_ACC_processed_data/data.csv", index=False)
    # train_df.to_csv("datasets/PPG_ACC_processed_data/train_data.csv", index=False)
    # test_df.to_csv("datasets/PPG_ACC_processed_data/test_data.csv", index=False)

    print("Processed data saved to data.csv, maybe also train_data.csv and test_data.csv")

    # return df, train_df, test_df
    return df

def mel_spectrogram(rawdata):
    """
    Compute the Mel spectrogram for the input raw data.
    """
    import librosa
    # Compute the Short-Time Fourier Transform (STFT)
    n_fft = 2048 # FFT window size
    hop_length = 64*4 # Number of samples between successive frames
    stft = np.abs(librosa.stft(rawdata, n_fft=2048, hop_length=512))

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(S=stft, n_mels=128)

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(os.listdir())
    if not os.path.exists("cognitive_music/datasets/PPG_ACC_processed_data/data.csv"):
        input("The processed data is not found. Press Enter to process the data...")
        df = prep_training_data()
    else:
        print("Processed data found. Loading...")
    df = pd.read_csv("cognitive_music\datasets\PPG_ACC_processed_data\data.csv")
        # train_df = pd.read_csv("PPG_ACC_processed_data/train_data.csv")
        # test_df = pd.read_csv("PPG_ACC_processed_data/test_data.csv")

    log_mel_speci = mel_spectrogram(np.array(df.ACCi.tolist()))
    log_mel_specj = mel_spectrogram(np.array(df.ACCj.tolist()))
    log_mel_speck = mel_spectrogram(np.array(df.ACCk.tolist()))
    # Save each spectrogram to appropriate file
    np.save("cognitive_music/datasets/PPG_ACC_processed_data/log_mel_speci.npy", log_mel_speci)
    np.save("cognitive_music/datasets/PPG_ACC_processed_data/log_mel_specj.npy", log_mel_specj)
    np.save("cognitive_music/datasets/PPG_ACC_processed_data/log_mel_speck.npy", log_mel_speck)

    import matplotlib.pyplot as plt

    # Display the first frame of log_mel_speci
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_speci[:, 0:300], aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram (First Frame) - ACCi')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    # Calculate the relative time spent on each activity
    activity_counts = df['ActivityDescr'].value_counts(normalize=True) * 100

    # Create a pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Relative Time Spent on Each Activity')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    


    plt.show()



