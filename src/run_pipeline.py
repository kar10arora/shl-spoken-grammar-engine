"""
Master End-to-End Execution Pipeline:
1. Loads acoustic features from data/train_audio_features.csv & data/test_audio_features.csv.
2. Extracts linguistic and grammatical features from data/train_transcripts.csv & data/test_transcripts.csv.
3. Merges multimodal feature spaces.
4. Executes 5-Fold Stratified Cross-Validation (Ridge Regressor + LightGBM Regressor).
5. Computes compulsory Training Data RMSE and OOF Validation metrics (Pearson r, RMSE, Spearman rho, MAE).
6. Outputs calibrated continuous predictions to data/submission.csv and prints full diagnostic summary.
"""

import os
import json
import numpy as np
import pandas as pd
from linguistic_features import extract_dataset_linguistic_features
from model_trainer import train_evaluate_pipeline

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"
OUTPUT_SUBMISSION = r"c:\Users\user\Desktop\SHL\submission.csv"
OUTPUT_METRICS_JSON = r"c:\Users\user\Desktop\SHL\data\evaluation_summary.json"


def main():
    print(f"\n{'='*70}")
    print("SHL Grammar Scoring Engine: Multimodal Training & Evaluation Pipeline")
    print(f"{'='*70}\n")

    # 1. Load Audio Features
    print("[1/4] Loading Acoustic Features...")
    train_audio_path = os.path.join(DATA_DIR, "train_audio_features.csv")
    test_audio_path = os.path.join(DATA_DIR, "test_audio_features.csv")
    if not os.path.exists(train_audio_path) or not os.path.exists(test_audio_path):
        raise FileNotFoundError("Audio features not found in data/. Run run_audio_extraction.py first.")

    train_audio_df = pd.read_csv(train_audio_path)
    test_audio_df = pd.read_csv(test_audio_path)
    print(f"Loaded train audio features: {train_audio_df.shape}, test audio features: {test_audio_df.shape}")

    # 2. Extract Linguistic Features from Transcripts
    print("\n[2/4] Extracting Linguistic & Grammatical Features from Transcripts...")
    train_trans_path = os.path.join(DATA_DIR, "train_transcripts.csv")
    test_trans_path = os.path.join(DATA_DIR, "test_transcripts.csv")
    if not os.path.exists(train_trans_path) or not os.path.exists(test_trans_path):
        raise FileNotFoundError("Transcripts not found in data/. Wait for run_transcription.py to finish.")

    train_trans_df = pd.read_csv(train_trans_path)
    test_trans_df = pd.read_csv(test_trans_path)

    # Merge audio duration info for speaking rate computation
    train_trans_df = train_trans_df.merge(train_audio_df[["filename", "duration_sec", "speech_duration_sec"]], on="filename", how="left")
    test_trans_df = test_trans_df.merge(test_audio_df[["filename", "duration_sec", "speech_duration_sec"]], on="filename", how="left")

    train_ling_df = extract_dataset_linguistic_features(train_trans_df)
    test_ling_df = extract_dataset_linguistic_features(test_trans_df)

    train_ling_df.to_csv(os.path.join(DATA_DIR, "train_linguistic_features.csv"), index=False)
    test_ling_df.to_csv(os.path.join(DATA_DIR, "test_linguistic_features.csv"), index=False)
    print("Linguistic features saved to data/")

    # 3. Merge Features into Unified Representation (Acoustic + Linguistic + J-Eval Judge)
    print("\n[3/4] Merging Acoustic, Linguistic, and J-Eval LLM Judge Modalities...")
    # Load J-Eval Judge scores
    train_judge_df = pd.read_csv(os.path.join(DATA_DIR, "train_judge_scores.csv"))
    test_judge_df = pd.read_csv(os.path.join(DATA_DIR, "test_judge_scores.csv"))
    judge_cols = ["filename", "judge_rubric_mos", "grammar_accuracy_score", "syntax_score", "sentence_completion_score"]

    ling_cols_to_merge = [c for c in train_ling_df.columns if c not in ["label", "duration_sec", "speech_duration_sec"]]
    
    train_full_df = train_audio_df.merge(train_ling_df[ling_cols_to_merge], on="filename", how="inner")
    train_full_df = train_full_df.merge(train_judge_df[judge_cols], on="filename", how="inner")

    test_full_df = test_audio_df.merge(test_ling_df[ling_cols_to_merge], on="filename", how="left")
    test_full_df = test_full_df.merge(test_judge_df[judge_cols], on="filename", how="left")
    for col in judge_cols[1:]:
        test_full_df[col] = test_full_df[col].fillna(3.0)

    print(f"Unified Train Matrix (with J-Eval Judge): {train_full_df.shape} (769 samples)")
    print(f"Unified Test Matrix  (with J-Eval Judge): {test_full_df.shape} (216 samples)")
    assert len(test_full_df) == 216, f"Expected 216 test rows, got {len(test_full_df)}"

    # 4. Train, Cross-Validate & Generate Predictions
    print("\n[4/4] Executing 5-Fold Stratified Regressor & Compulsory Metric Benchmarking...")
    sub_df, results_summary = train_evaluate_pipeline(train_full_df, test_full_df, n_splits=5, random_state=42)

    # Save final submission
    sub_df.to_csv(OUTPUT_SUBMISSION, index=False)
    sub_df.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print(f"\n[OK] Final Submission successfully saved to: {OUTPUT_SUBMISSION}")
    print(f"Total test predictions: {len(sub_df)} rows")
    print(sub_df.head(10))

    # Save evaluation summary JSON
    with open(OUTPUT_METRICS_JSON, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[OK] Diagnostic evaluation metrics saved to: {OUTPUT_METRICS_JSON}")


if __name__ == "__main__":
    main()
