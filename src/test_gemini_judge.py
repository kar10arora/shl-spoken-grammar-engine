import os
import json
from dotenv import load_dotenv
load_dotenv('.env')

import google.generativeai as genai
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Test with gemini-flash-latest / gemini-2.5-flash
model = genai.GenerativeModel(
    model_name='gemini-flash-latest',
    generation_config={"response_mime_type": "application/json"}
)

sample_text = "Well, the playground is very, very beautiful and fun. There are, there are places like the neighborhood around the street, the street is really interesting, it has to be like, um, it's more like a park with trees."

prompt = f"""You are the Lead Linguistic Assessor for the SHL Spoken Grammar competition.
Evaluate the spoken English grammar of this transcript based on the official rubric (Score 1 to 5):
\"\"\"{sample_text}\"\"\"

Return a valid JSON object with:
{{
  "syntax_score": float (1.0 to 5.0),
  "grammar_accuracy_score": float (1.0 to 5.0),
  "sentence_completion_score": float (1.0 to 5.0),
  "judge_rubric_mos": float (1.0 to 5.0 continuous score),
  "reasoning": string
}}"""

print("Sending request to Gemini Judge...")
res = model.generate_content(prompt)
print("Raw Response text:")
print(res.text)

parsed = json.loads(res.text)
print("Parsed JSON successfully:")
print(json.dumps(parsed, indent=2))
