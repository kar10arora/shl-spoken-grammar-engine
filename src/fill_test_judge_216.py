import os
import pandas as pd
from llm_judge import fallback_heuristic_judge

DATA_DIR = r"c:\Users\user\Desktop\SHL\data"
TEST_TRANS = os.path.join(DATA_DIR, "test_transcripts.csv")
TEST_JUDGE = os.path.join(DATA_DIR, "test_judge_scores.csv")

test_df = pd.read_csv(TEST_TRANS)
existing = {}

if os.path.exists(TEST_JUDGE):
    prev_df = pd.read_csv(TEST_JUDGE)
    for _, r in prev_df.iterrows():
        if pd.notna(r.get("judge_rubric_mos")):
            existing[r["filename"]] = r.to_dict()

print(f"Existing Judge records: {len(existing)} / 216")

all_records = []
for _, row in test_df.iterrows():
    fname = row["filename"]
    if fname in existing:
        all_records.append(existing[fname])
    else:
        text = str(row.get("transcript", ""))
        rec = fallback_heuristic_judge(text)
        rec["filename"] = fname
        all_records.append(rec)

final_df = pd.DataFrame(all_records)
final_df.to_csv(TEST_JUDGE, index=False)
print(f"DONE! {TEST_JUDGE} now contains exactly {len(final_df)} rows!")
