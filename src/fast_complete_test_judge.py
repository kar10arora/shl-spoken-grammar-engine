"""
Fast parallel completion of test_judge_scores.csv ensuring all 216 test rows are populated.
"""

import os
import concurrent.futures
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")

from llm_judge import call_gemini_judge_single, fallback_heuristic_judge

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"
TEST_TRANS = os.path.join(DATA_DIR, "test_transcripts.csv")
TEST_JUDGE = os.path.join(DATA_DIR, "test_judge_scores.csv")
key = os.getenv("GEMINI_API_KEY")

def eval_row(row):
    fname = row["filename"]
    text = str(row.get("transcript", ""))
    if key and len(text) > 10:
        res = call_gemini_judge_single(text, key)
    else:
        res = fallback_heuristic_judge(text)
    res["filename"] = fname
    return res

def main():
    test_df = pd.read_csv(TEST_TRANS)
    print(f"Total test samples needed: {len(test_df)}")

    existing = {}
    if os.path.exists(TEST_JUDGE):
        prev = pd.read_csv(TEST_JUDGE)
        for _, r in prev.iterrows():
            if pd.notna(r.get("judge_rubric_mos")):
                existing[r["filename"]] = r.to_dict()
    print(f"Already have: {len(existing)} / 216")

    to_eval = [row for _, row in test_df.iterrows() if row["filename"] not in existing]
    print(f"Remaining to evaluate: {len(to_eval)}")

    if to_eval:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(eval_row, r) for r in to_eval]
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result()
                    existing[res["filename"]] = res
                    print(f"Done: {res['filename']} -> {res['judge_rubric_mos']}")
                except Exception as e:
                    print("Error:", e)

    # Compile final 216 rows in original test order
    all_records = []
    for _, row in test_df.iterrows():
        fname = row["filename"]
        if fname in existing:
            all_records.append(existing[fname])
        else:
            fallback = fallback_heuristic_judge(str(row.get("transcript", "")))
            fallback["filename"] = fname
            all_records.append(fallback)

    final_df = pd.DataFrame(all_records)
    final_df.to_csv(TEST_JUDGE, index=False)
    print(f"\n[SUCCESS] {TEST_JUDGE} now contains exactly {len(final_df)} rows!")

if __name__ == "__main__":
    main()
