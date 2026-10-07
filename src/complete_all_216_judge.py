"""
Ensures data/test_judge_scores.csv contains exactly all 216 test samples.
Fills any missing sample with Gemini / calibrated rubric judge immediately.
"""

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")

from llm_judge import call_gemini_judge_single, fallback_heuristic_judge

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"
TEST_TRANS = os.path.join(DATA_DIR, "test_transcripts.csv")
TEST_JUDGE = os.path.join(DATA_DIR, "test_judge_scores.csv")
key = os.getenv("GEMINI_API_KEY")

def main():
    test_df = pd.read_csv(TEST_TRANS)
    print(f"Target: ensure all {len(test_df)} test samples exist in {TEST_JUDGE}")

    existing = {}
    if os.path.exists(TEST_JUDGE):
        prev = pd.read_csv(TEST_JUDGE)
        for _, r in prev.iterrows():
            if pd.notna(r.get("judge_rubric_mos")):
                existing[r["filename"]] = r.to_dict()
        print(f"Loaded {len(existing)} existing judge verdicts.")

    records = []
    missing_count = 0
    for _, row in test_df.iterrows():
        fname = row["filename"]
        if fname in existing:
            rec = existing[fname]
        else:
            missing_count += 1
            text = str(row.get("transcript", ""))
            if key and len(text) > 10:
                rec = call_gemini_judge_single(text, key)
            else:
                rec = fallback_heuristic_judge(text)
            rec["filename"] = fname
            existing[fname] = rec
            print(f"Filled missing [{missing_count}] {fname} -> {rec['judge_rubric_mos']}")

        records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(TEST_JUDGE, index=False)
    print(f"\nSUCCESS! {TEST_JUDGE} now contains exactly {len(out_df)} rows (expected: 216)!")

if __name__ == "__main__":
    main()
