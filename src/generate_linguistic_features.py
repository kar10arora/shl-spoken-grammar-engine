"""
Generates linguistic features for train transcripts and verifies all 4 CSVs in data/.
"""

import os
import pandas as pd
from linguistic_features import extract_dataset_linguistic_features

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"

def main():
    print("Generating train linguistic features...")
    train_audio = pd.read_csv(os.path.join(DATA_DIR, "train_audio_features.csv"))
    train_trans = pd.read_csv(os.path.join(DATA_DIR, "train_transcripts.csv"))
    train_trans = train_trans.merge(train_audio[["filename", "duration_sec", "speech_duration_sec"]], on="filename", how="left")

    train_ling = extract_dataset_linguistic_features(train_trans)
    out_path = os.path.join(DATA_DIR, "train_linguistic_features.csv")
    train_ling.to_csv(out_path, index=False)
    print(f"Generated {out_path} successfully! Shape: {train_ling.shape}")

    # Check all files
    for name in ["train_audio_features.csv", "test_audio_features.csv", "train_linguistic_features.csv", "test_linguistic_features.csv"]:
        p = os.path.join(DATA_DIR, name)
        print(f"  - {name}: exists={os.path.exists(p)}, rows={len(pd.read_csv(p)) if os.path.exists(p) else 0}")

if __name__ == "__main__":
    main()
