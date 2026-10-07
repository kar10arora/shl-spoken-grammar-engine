import os
import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env")

from llm_judge import call_gemini_judge_single

trans_df = pd.read_csv("data/train_transcripts.csv")
key = os.getenv("GEMINI_API_KEY")

print("Testing 3 live Gemini Judge calls on real transcripts...")
for idx in range(3):
    row = trans_df.iloc[idx]
    fname = row["filename"]
    t0 = time.time()
    verdict = call_gemini_judge_single(row["transcript"], key)
    print(f"\nSample {idx+1} ({fname}) [Elapsed: {time.time()-t0:.2f}s]:")
    print(f" - Syntax Score: {verdict['syntax_score']}")
    print(f" - Grammar Score: {verdict['grammar_accuracy_score']}")
    print(f" - Verdict MOS: {verdict['judge_rubric_mos']}")
    print(f" - Reasoning: {verdict['reasoning']}")
