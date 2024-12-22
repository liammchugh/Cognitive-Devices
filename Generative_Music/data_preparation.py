"""
This script downloads or loads commonly used public datasets for:
- Music/audio data with labels (e.g., MagnaTagATune or FMA).
- Activity/accelerometer data (e.g., UCI HAR Dataset).

It then preprocesses and stores them in a directory structure for subsequent training.
"""

import os
import requests
import zipfile
import tarfile
import shutil
import pandas as pd
import numpy as np

def download_uci_har_data(destination_dir="data/uci_har"):
    """
    Download the UCI HAR Dataset (if not already downloaded).
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_filename = os.path.join(destination_dir, "UCI_HAR_Dataset.zip")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if not os.path.exists(zip_filename):
        print("Downloading UCI HAR Dataset...")
        r = requests.get(url, stream=True)
        with open(zip_filename, 'wb') as f:
            f.write(r.content)
    else:
        print("UCI HAR Dataset zip already exists.")

    # Unzip
    extract_dir = os.path.join(destination_dir, "UCI_HAR_Dataset")
    if not os.path.exists(extract_dir):
        print("Extracting UCI HAR Dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as z:
            z.extractall(destination_dir)

def download_magnatagatune_data(destination_dir="data/magnatagatune"):
    """
    Placeholder for MagnaTagATune or similar. 
    Real dataset requires licensing or custom scripts.
    For demonstration, we create a dummy folder structure.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        # In a real scenario, implement the actual data download script
        # For now, we just put a placeholder text file.
        with open(os.path.join(destination_dir, "placeholder.txt"), "w") as f:
            f.write("MagnaTagATune data placeholder.\n")
    print("MagnaTagATune placeholder data prepared.")

def preprocess_uci_har_data(source_dir="data/uci_har/UCI_HAR_Dataset", output_dir="data/processed"):
    """
    Load and preprocess UCI HAR data into a convenient format (e.g., CSV files or NumPy arrays).
    """
    # Check if the dataset exists
    if not os.path.exists(source_dir):
        print("UCI HAR raw data not found. Skipping.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Example: combine training/test data
    # For a real pipeline, parse the "train" and "test" subfolders
    # to create a single consolidated dataset.
    # We'll create a dummy output for demonstration.
    print("Preprocessing UCI HAR data...")
    dummy_activity_data = np.random.randn(1000, 3)  # e.g. 3-axis accelerometer
    dummy_labels = np.random.randint(0, 6, size=(1000,))
    
    np.save(os.path.join(output_dir, "accelerometer_features.npy"), dummy_activity_data)
    np.save(os.path.join(output_dir, "activity_labels.npy"), dummy_labels)
    print(f"Saved preprocessed accelerometer features to {output_dir}.")

def preprocess_magnatagatune_data(source_dir="data/magnatagatune", output_dir="data/processed"):
    """
    Placeholder for real audio feature extraction.
    In practice, you'd read audio files, compute mel-spectrograms, 
    gather tags, etc., and store them.
    """
    if not os.path.exists(source_dir):
        print("MagnaTagATune data not found. Skipping.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Preprocessing MagnaTagATune data (placeholder).")
    # Just a dummy example storing random spectrogram data
    dummy_spectrograms = np.random.randn(100, 128, 128)  # (num_samples, freq_bins, time_frames)
    dummy_tags = np.random.randint(0, 2, size=(100, 10)) # e.g., 10 possible tags

    np.save(os.path.join(output_dir, "spectrograms.npy"), dummy_spectrograms)
    np.save(os.path.join(output_dir, "tags.npy"), dummy_tags)
    print(f"Saved preprocessed music data to {output_dir}.")

def main():
    # Step 1: Download datasets
    download_uci_har_data()
    download_magnatagatune_data()

    # Step 2: Preprocess data
    preprocess_uci_har_data()
    preprocess_magnatagatune_data()

    print("Data preparation completed.")

if __name__ == "__main__":
    main()
