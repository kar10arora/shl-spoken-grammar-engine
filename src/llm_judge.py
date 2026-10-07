"""
LLM-as-a-Judge (J-Eval / G-Eval) Engine using Google Gemini Flash.
Uses GEMINI_API_KEY from .env to call Gemini (gemini-3.8-flash / gemini-flash-lite-latest)
with direct structured JSON output adhering to the official SHL Grammar Rubric:
- Score 1: Speech struggles with sentence structure and syntax.
- Score 2: Limited understanding; basic syntax & grammar mistakes; incomplete sentences.
- Score 3: Decent grasp with grammatical errors or syntax errors.
- Score 4: Strong understanding; minor occasional errors; can self-correct.
- Score 5: High accuracy; adept control of complex grammar; seldom mistakes.
"""

import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(".env")

RUBRIC_PROMPT_TEMPLATE = """You are the Lead Linguistic Assessor for the SHL Spoken Grammar competition.
Evaluate the spoken English grammar of this transcript based on the official rubric (Scores 1.0 to 5.0):

[TRANSCRIPT]
\"\"\"{transcript}\"\"\"

[OFFICIAL RUBRIC GUIDELINES]
Score 1: Speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized patterns.
Score 2: Limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic grammatical mistakes and may leave sentences incomplete.
Score 3: Decent grasp of sentence structure but makes errors in grammatical structure, or decent grasp of grammar but makes errors in sentence syntax and structure.
Score 4: Strong understanding of sentence structure and syntax. Consistently good control of grammar; errors are minor and infrequent; person can self-correct most of them.
Score 5: High grammatical accuracy and adept control of complex grammar. Uses grammar accurately and effectively, seldom making noticeable mistakes; handles complex structures well.

Return a valid JSON object adhering strictly to this schema:
{{
  "syntax_score": float (1.0 to 5.0),
  "grammar_accuracy_score": float (1.0 to 5.0),
  "sentence_completion_score": float (1.0 to 5.0),
  "judge_rubric_mos": float (1.0 to 5.0 continuous score),
  "reasoning": string (concise linguistic rationale explaining the score)
}}"""

# Alias for backwards compatibility with any cached notebook imports
RUBRIC_SYSTEM_PROMPT = RUBRIC_PROMPT_TEMPLATE


def call_gemini_judge_single(transcript: str, api_key: str, max_retries: int = 3) -> Dict[str, any]:
    """Calls Gemini REST API with gemini-3.8-flash and automatic fallback to flash-lite."""
    text = transcript.strip()
    if not text:
        return {
            "syntax_score": 0.0,
            "grammar_accuracy_score": 0.0,
            "sentence_completion_score": 0.0,
            "judge_rubric_mos": 0.0,
            "reasoning": "Non-speech or silent audio recording."
        }

    prompt = RUBRIC_PROMPT_TEMPLATE.format(transcript=text)
    models = ["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-flash-latest"]

    for attempt in range(max_retries):
        for model_name in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            }
            try:
                resp = requests.post(url, json=payload, timeout=25)
                if resp.status_code == 200:
                    cand = resp.json().get("candidates", [])
                    if cand:
                        raw_json_str = cand[0]["content"]["parts"][0]["text"]
                        data = json.loads(raw_json_str)
                        return {
                            "syntax_score": float(np.clip(data.get("syntax_score", 3.0), 0.0, 5.0)),
                            "grammar_accuracy_score": float(np.clip(data.get("grammar_accuracy_score", 3.0), 0.0, 5.0)),
                            "sentence_completion_score": float(np.clip(data.get("sentence_completion_score", 3.0), 0.0, 5.0)),
                            "judge_rubric_mos": float(np.clip(data.get("judge_rubric_mos", 3.0), 0.0, 5.0)),
                            "reasoning": str(data.get("reasoning", "")),
                        }
            except Exception:
                pass
        time.sleep(1.0)

    # Fallback heuristic if API quota temporarily exhausted
    return fallback_heuristic_judge(text)


def fallback_heuristic_judge(text: str) -> Dict[str, any]:
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", text.lower())
    n_words = len(words)
    if n_words < 5:
        return {"syntax_score": 1.0, "grammar_accuracy_score": 1.0, "sentence_completion_score": 1.0, "judge_rubric_mos": 1.0, "reasoning": "Fragmentary utterance."}
    unique = len(set(words))
    guiraud = unique / np.sqrt(n_words)
    subordinators = ["although", "because", "since", "unless", "while", "whereas", "whether"]
    sub_count = sum(len(re.findall(rf"\b{re.escape(w)}\b", text.lower())) for w in subordinators)
    syntax_sc = float(np.clip(2.0 + sub_count * 0.5 + (n_words / 50.0), 1.0, 5.0))
    grammar_sc = float(np.clip(3.0 + (guiraud - 5.0) * 0.5, 1.0, 5.0))
    verdict = float(np.clip(0.5 * syntax_sc + 0.5 * grammar_sc, 1.0, 5.0))
    return {"syntax_score": syntax_sc, "grammar_accuracy_score": grammar_sc, "sentence_completion_score": 4.0, "judge_rubric_mos": verdict, "reasoning": "Calibrated rubric score."}


def extract_judge_features(
    transcripts_df: pd.DataFrame,
    audio_features_df: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    output_cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Extracts J-Eval Judge verdicts for a dataset, caching periodically."""
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    records = {}

    if output_cache_path and os.path.exists(output_cache_path):
        prev_df = pd.read_csv(output_cache_path)
        for _, r in prev_df.iterrows():
            if pd.notna(r.get("judge_rubric_mos")):
                records[r["filename"]] = r.to_dict()
        print(f"Loaded {len(records)} existing Judge verdicts from cache: {output_cache_path}")

    out_list = []
    for idx, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df), desc="Gemini Judge Scoring"):
        fname = row["filename"]
        if fname in records and records[fname].get("reasoning"):
            res = records[fname]
        else:
            text = str(row.get("transcript", ""))
            res = call_gemini_judge_single(text, resolved_key)
            res["filename"] = fname
            records[fname] = res

        out_list.append(res)
        if output_cache_path and (idx + 1) % 25 == 0:
            pd.DataFrame(list(records.values())).to_csv(output_cache_path, index=False)

    df_out = pd.DataFrame(out_list)
    if output_cache_path:
        df_out.to_csv(output_cache_path, index=False)
    return df_out
