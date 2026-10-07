"""
Generates live Gemini LLM Judge evaluations for all 216 test transcripts.
Saves checkpoints periodically to data/test_judge_scores.csv.
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")

from llm_judge import call_gemini_judge_single

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"
OUTPUT_CSV = os.path.join(DATA_DIR, "test_judge_scores.csv")
key = os.getenv("GEMINI_API_KEY")

def main():
    test_trans = pd.read_csv(os.path.join(DATA_DIR, "test_transcripts.csv"))
    print(f"Running live Gemini LLM Judge on {len(test_trans)} test samples...")

    existing = {}
    records = []
    print(f"Executing fresh live Gemini LLM calls on {len(test_trans)} test transcripts...")
    for idx, row in test_trans.iterrows():
        fname = row["filename"]
        if fname in existing:
            verdict = existing[fname]
        else:
            verdict = call_gemini_judge_single(str(row.get("transcript", "")), key)
            verdict["filename"] = fname
            existing[fname] = verdict
            print(f"[{idx+1}/{len(test_trans)}] {fname} -> Judge MOS: {verdict['judge_rubric_mos']} | {verdict['reasoning'][:90]}...")
            time.sleep(0.5)

        records.append(verdict)
        if (idx + 1) % 20 == 0 or (idx + 1) == len(test_trans):
            pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Completed all {len(df_out)} test Judge evaluations -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
