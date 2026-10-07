"""
Script to extract audio signal features for train and test splits and save to data/.
"""

import os
import pandas as pd
from audio_features import extract_dataset_audio_features

DATASET_DIR = r"c:\Users\user\Desktop\SHL\shl-hiring-assessment-2026\Dataset_Final"
OUTPUT_DIR = r"c:\Users\user\Desktop\SHL\data"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Train audio features
    train_csv = os.path.join(DATASET_DIR, "train.csv")
    train_audio_dir = os.path.join(DATASET_DIR, "train")
    train_df = pd.read_csv(train_csv)
    print(f"Extracting audio features for {len(train_df)} train samples...")
    train_feat_df = extract_dataset_audio_features(train_audio_dir, train_df, "filename")
    train_feat_df["label"] = train_df["label"]
    train_out_path = os.path.join(OUTPUT_DIR, "train_audio_features.csv")
    train_feat_df.to_csv(train_out_path, index=False)
    print(f"Saved train audio features to: {train_out_path}")

    # 2. Test audio features
    test_audio_dir = os.path.join(DATASET_DIR, "test")
    test_files = sorted(os.listdir(test_audio_dir))
    test_df = pd.DataFrame({"filename": test_files})
    print(f"Extracting audio features for {len(test_df)} test samples...")
    test_feat_df = extract_dataset_audio_features(test_audio_dir, test_df, "filename")
    test_out_path = os.path.join(OUTPUT_DIR, "test_audio_features.csv")
    test_feat_df.to_csv(test_out_path, index=False)
    print(f"Saved test audio features to: {test_out_path}")

    print("Signal feature extraction completed successfully!")

if __name__ == "__main__":
    main()
