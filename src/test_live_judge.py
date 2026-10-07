import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(".env")
key = os.getenv("GEMINI_API_KEY")

sample_transcript = """My favorite hobby is about playing shuttle. It's my one of the most enjoyable hobby. I used to play every day with shuttle. I enjoy practicing with my own. It's also my friends because it makes me more and more competitive day by day. It increases the sporty skills and healthy fit towards my mental and physical happiness. It makes me fit."""

prompt = f"""You are the Lead Linguistic Assessor for the SHL Spoken Grammar challenge.
Evaluate this spoken transcript based on the official rubric (Score 1 to 5):
\"\"\"{sample_transcript}\"\"\"

Return valid JSON with:
{{
  "syntax_score": float (1.0 to 5.0),
  "grammar_accuracy_score": float (1.0 to 5.0),
  "sentence_completion_score": float (1.0 to 5.0),
  "judge_rubric_mos": float (1.0 to 5.0 continuous score),
  "reasoning": string
}}"""

models_to_try = ["gemini-3.8-flash", "gemini-flash-lite-latest", "gemini-flash-latest"]
success = False

for m in models_to_try:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"},
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            print(f"SUCCESS with {m}!")
            res_json = r.json()
            text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            verdict = json.loads(text)
            print("Gemini Judge Verdict:")
            print(json.dumps(verdict, indent=2))
            success = True
            break
        else:
            print(f"{m} returned status {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"{m} error: {e}")

if not success:
    print("Could not complete call with tried models.")
