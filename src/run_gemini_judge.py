"""
Script to run the live Gemini LLM-as-a-Judge on test and train transcripts.
"""

import os
import pandas as pd
from llm_judge import extract_judge_features

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"

def main():
    print("Executing Gemini LLM Judge on test transcripts (216 samples)...")
    test_trans = pd.read_csv(os.path.join(DATA_DIR, "test_transcripts.csv"))
    test_out_csv = os.path.join(DATA_DIR, "test_judge_scores.csv")
    test_judge_df = extract_judge_features(test_trans, output_cache_path=test_out_csv)
    print("Test Judge Completed! Sample verdicts:")
    print(test_judge_df[["filename", "syntax_score", "grammar_accuracy_score", "judge_rubric_mos"]].head())

    print("\nExecuting Gemini LLM Judge on train transcripts (769 samples)...")
    train_trans = pd.read_csv(os.path.join(DATA_DIR, "train_transcripts.csv"))
    train_out_csv = os.path.join(DATA_DIR, "train_judge_scores.csv")
    train_judge_df = extract_judge_features(train_trans, output_cache_path=train_out_csv)
    print("Train Judge Completed!")

if __name__ == "__main__":
    main()
