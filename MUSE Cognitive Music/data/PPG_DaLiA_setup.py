import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()             # …/your_script.py
PROJECT_ROOT = THIS_FILE.parents[1]              # adjust depth as needed
DATA_DIR = PROJECT_ROOT / "data" / "ppg+dalia" / "data" / "PPG_FieldStudy"
PROCESSED_DIR = PROJECT_ROOT / "data" / "PPG_ACC_processed_data"

# make sure the path is import-safe for your own repos/modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ACTIVITY_LABELS = {
    0: "Transition",            # transient / unlabeled
    1: "Sitting and Reading",
    2: "Climbing Stairs",
    3: "Playing Table Soccer",
    4: "Cycling Outdoors",
    5: "Driving a Car",
    6: "Lunch Break",
    7: "Walking",
    8: "Working at Desk",
}

def prep_training_data(
    data_dir: Path = DATA_DIR,
    processed_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    """
    Load the raw PPG-DaLiA files located in *data_dir*, resample /
    interpolate the signals so everything aligns to the PPG timeline,
    attach per-subject metadata, and write a single CSV to
    *processed_dir / 'data.csv'*.

    Returns
    -------
    pd.DataFrame
        One row per PPG sample, with ACC-xyz, heart-rate label,
        integer activity-ID, text activity description, and subject
        demographics.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"PPG-DaLiA directory not found at {data_dir}. "
            "Download the dataset and/or point DATA_DIR to it."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Accumulators
    # ----------------------------------------------------------------
    ppg_all, acc_all, hr_all, act_all = [], [], [], []
    subj_id_all, age_all, gender_all = [], [], []
    height_all, weight_all, skin_all, sport_all = [], [], [], []

    for subj in range(1, 16):                       # S1 … S15
        pkl_path  = data_dir / f"S{subj}" / f"S{subj}.pkl"
        csv_path  = data_dir / f"S{subj}" / f"S{subj}_quest.csv"

        if not pkl_path.exists():
            print(f"[WARN] {pkl_path} missing – skipping subject {subj}")
            continue

        # --------------------- signals ------------------------------
        with pkl_path.open("rb") as fh:
            raw = pickle.load(fh, encoding="latin1")

        ppg = np.asarray(raw["signal"]["wrist"]["BVP"])      # 64 Hz
        acc = np.asarray(raw["signal"]["wrist"]["ACC"])      # 32 Hz
        hr  = np.asarray(raw["label"])                       #   1 Hz
        act = np.asarray(raw["activity"])                    #   1 Hz

        # --------------------- resample ACC to 64 Hz ---------------
        acc_interp = np.zeros((ppg.shape[0], acc.shape[1]))
        for i in range(acc.shape[1]):
            acc_interp[:, i] = np.interp(
                np.linspace(0, len(acc) - 1, num=len(ppg)),
                np.arange(len(acc)),
                acc[:, i]
            )
        # --------------------- resample HR & activity --------------
        hr_interp = np.interp(
            np.linspace(0, len(hr) - 1, num=len(ppg)),
            np.arange(len(hr)),
            hr
        )

        act_interp = np.round(np.interp(
            np.linspace(0, len(act) - 1, num=len(ppg)),
            np.arange(len(act)),
            act.flatten()
        )).astype(int)

        # --------------------- metadata ----------------------------
        # defaults in case the CSV is missing / malformed
        meta = {
            "SUBJECT_ID": f"S{subj}",
            "AGE":        -1,
            "GENDER":     "Unknown",
            "HEIGHT":     -1,
            "WEIGHT":     -1,
            "SKIN":       -1,
            "SPORT":      -1,
        }
        try:
            with csv_path.open() as fh:
                meta.update({
                    k.strip(): v.strip()
                    for k, v in
                    (line.lstrip("#").split(",", 1) for line in fh if line.strip())
                })
        except FileNotFoundError:
            print(f"[WARN] {csv_path} missing – demographics defaulted")

        # --------------------- stack subject -----------------------
        n = len(ppg)
        ppg_all.append(ppg)
        acc_all.append(acc_interp)
        hr_all.append(hr_interp)
        act_all.append(act_interp)

        subj_id_all.extend([meta["SUBJECT_ID"]]*n)
        age_all.extend([int(meta["AGE"])]*n)
        gender_all.extend([meta["GENDER"]]*n)
        height_all.extend([int(meta["HEIGHT"])]*n)
        weight_all.extend([int(meta["WEIGHT"])]*n)
        skin_all.extend([int(meta["SKIN"])]*n)
        sport_all.extend([int(meta["SPORT"])]*n)

    # ----------------------------------------------------------------
    # Concatenate across subjects
    # ----------------------------------------------------------------
    ppg_all = np.concatenate(ppg_all, axis=0)
    acc_all = np.concatenate(acc_all, axis=0)
    hr_all  = np.concatenate(hr_all,  axis=0)
    act_all = np.concatenate(act_all, axis=0)

    print(f"→ PPG     : {ppg_all.shape}")
    print(f"→ ACC xyz : {acc_all.shape}")
    print(f"→ HR      : {hr_all.shape}")
    print(f"→ Activity: {act_all.shape}")

    # textual activity labels
    act_descr = [_ACTIVITY_LABELS[i] for i in act_all]

    df = pd.DataFrame({
        "PPG":      list(ppg_all),
        "ACCi":     list(acc_all[:, 0]),
        "ACCj":     list(acc_all[:, 1]),
        "ACCk":     list(acc_all[:, 2]),
        "HeartRate": hr_all,
        "Activity":  act_all,
        "ActivityDescr": act_descr,
        "SubjectID": subj_id_all,
        "Age":       age_all,
        "Gender":    gender_all,
        "Height":    height_all,
        "Weight":    weight_all,
        "SkinType":  skin_all,
        "SportLevel": sport_all,
    })

    # optional split
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # ----------------------------------------------------------------
    # Persist to disk
    # ----------------------------------------------------------------
    # csv_path = processed_dir / "data.csv"
    parquet_path = processed_dir / "data.parquet"
    df.to_parquet(parquet_path)   # ~4-5× smaller, faster I/O
    print(f"Saved full dataset → {parquet_path}")

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

    if not (PROCESSED_DIR / "data.csv").exists():
        input("The processed data is not found. Press Enter to process the data...")
        df = prep_training_data()
    else:
        print("Processed data found. Loading...")
    csv_path = PROCESSED_DIR / "data.csv"
    df = pd.read_csv(csv_path)

    print(f"Performing spectrogram processing on {len(df)} samples...")
    log_mel_speci = mel_spectrogram(np.array(df.ACCi.tolist()))
    log_mel_specj = mel_spectrogram(np.array(df.ACCj.tolist()))
    log_mel_speck = mel_spectrogram(np.array(df.ACCk.tolist()))
    # Save each spectrogram to appropriate file
    np.save(PROCESSED_DIR / "log_mel_speci.npy", log_mel_speci)
    np.save(PROCESSED_DIR / "log_mel_specj.npy", log_mel_specj)
    np.save(PROCESSED_DIR / "log_mel_speck.npy", log_mel_speck)

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



