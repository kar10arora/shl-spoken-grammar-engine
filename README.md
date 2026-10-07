# SHL Spoken Grammar Scoring Engine
### Multimodal Acoustic-Linguistic Architecture with Calibrated LLM-as-a-Judge (J-Eval)
**Role:** Research Engineer Assessment 2026  
**Competition:** Private Kaggle Challenge — Grammar Scoring for Spoken Data (`shl-hiring-assessment-2026`)  
**Primary Metrics:** Pearson Correlation Coefficient ($r$) & Root Mean Squared Error ($\text{RMSE}$)  
**Leaderboard Score:** **`0.6077`** | **Compulsory Training RMSE:** **`0.6367`**

---

## 1. Executive Summary

This repository implements a production-grade, multimodal automated scoring engine designed to predict continuous **Mean Opinion Score (MOS)** grammar ratings ($0.0 \text{ to } 5.0$) from 45–60 second spoken English audio responses. 

The engine aligns directly with the official human assessment rubric:
* **Score 1:** Severe struggle with sentence structure and syntax; limited control over simple patterns.
* **Score 2:** Limited understanding of syntax; frequent basic grammatical errors; incomplete thoughts.
* **Score 3:** Decent grasp of sentence structure with grammatical errors, or decent grammar with syntax errors.
* **Score 4:** Strong understanding of syntax and grammar; minor, occasional errors; effective self-repair.
* **Score 5:** High grammatical accuracy; adept control of complex language structures; fluent self-correction.

---

## 2. Key Results & Performance Benchmark

| Evaluation Split | Pearson Correlation ($r$) | RMSE | Spearman ($\rho$) | MAE |
| :--- | :---: | :---: | :---: | :---: |
| **TRAINING DATA (Compulsory Requirement)** | **`0.8657`** | **`0.6367`** | **`0.8022`** | **`0.5148`** |
| **OUT-OF-FOLD (OOF) 5-Fold Cross-Validation** | **`0.7572`** | **`0.8098`** | **`0.6133`** | **`0.6559`** |
| **KAGGLE PUBLIC LEADERBOARD** | — | **`0.6077`** | — | — |

> **Mandatory Deliverable Notice:** The training RMSE (`0.6367`) is explicitly outputted in the Jupyter Notebook [`SHL_Grammar_Scoring_Engine.ipynb`](SHL_Grammar_Scoring_Engine.ipynb) and cross-validation summaries.

---

## 3. System Architecture & Methodology

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      RAW SPOKEN AUDIO (.wav, 45-60s)                      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
   ┌────────────────────┐   ┌──────────────────┐   ┌────────────────────┐
   │ Acoustic Grounding │   │  Verbatim ASR    │   │  Gemini LLM Judge  │
   │  Signal Analysis   │   │ (Whisper Int8)   │   │  (J-Eval / Rubric) │
   └──────────┬─────────┘   └─────────┬────────┘   └──────────┬─────────┘
              │                       │                       │
              │ Physical RMS, Silence │ Syntactic & Lexical   │ Rubric MOS, Syntax,│
              │ Energy Variance, ZCR  │ Disfluency Statistics │ Grammar Acc. Scores│
              │                       │                       │                    │
              └───────────────────────┼───────────────────────┘
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │       Unified Multimodal Feature Matrix         │
             │       (Acoustic + Linguistic + LLM Judge)       │
             └────────────────────────┬────────────────────────┘
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │     5-Fold Stratified Regressor Ensemble        │
             │    20% Ridge Regressor + 80% LightGBM (L2)      │
             └────────────────────────┬────────────────────────┘
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │        Continuous MOS Prediction [0.0, 5.0]     │
             │       Output: submission.csv (216 test rows)    │
             └─────────────────────────────────────────────────┘
```

### 3.1 The Multimodal Dilemma: Why Audio + Transcript + LLM?
1. **The ASR Auto-Correction Trap:** Modern Speech-to-Text architectures (e.g., standard Whisper) implicitly smooth out grammatical errors, false starts, and stutters. To preserve grammatical fidelity, we transcribe using `temperature=0.0`, `condition_on_previous_text=False`, and custom VAD chunking to capture verbatim disfluencies.
2. **The 0.0 MOS Anomaly (Acoustic Grounding):** In `train.csv`, 37 recordings have ground truth MOS `0.0` (non-speech, silence, or hardware corruption). Discarding acoustic signals would cause catastrophic failures on silent recordings. Physical RMS energy, silence ratios, and zero-crossing rates reliably isolate these non-speech samples.
3. **Calibrated LLM Judge (J-Eval):** While an LLM judge evaluates complex human rubrics accurately, raw LLM outputs suffer from central tendency bias and integer discretization. We extract Gemini's qualitative rubric scores and feed them into a calibrated **Ridge + LightGBM** ensemble directly optimized for **Pearson $r$** and **RMSE**.

---

## 4. Repository Structure

```text
├── SHL_Grammar_Scoring_Engine.ipynb   # Publication-grade Jupyter Notebook with full EDA, models, and plots
├── IMPLEMENTATION_PLAN.md             # Formal engineering & architecture specification
├── README.md                          # Repository documentation & benchmark report
├── requirements.txt                   # Exact pinned dependencies
├── .gitignore                         # Excludes heavy raw audio, virtualenvs, and .env keys
├── submission.csv                     # Final Kaggle submission (216 test predictions, Score: 0.6077)
│
├── src/                               # Modular Python source code
│   ├── audio_features.py              # Physical acoustics (RMS, silence ratios, zero crossing)
│   ├── transcribe.py                  # Verbatim ASR transcription with disfluency preservation
│   ├── linguistic_features.py         # 22 syntactic, lexical, and fluency feature extractors
│   ├── llm_judge.py                   # Official rubric prompt & Gemini REST client
│   ├── model_trainer.py               # 5-Fold Stratified Ridge + LightGBM trainer
│   ├── metrics.py                     # Pearson r, RMSE, Spearman rho, and MAE benchmark suite
│   ├── run_pipeline.py                # End-to-end execution script
│   └── create_notebook.py             # Automated notebook generator
│
└── data/                              # Extracted features and metadata
    ├── train_audio_features.csv       # Precomputed acoustic features (769 train samples)
    ├── test_audio_features.csv        # Precomputed acoustic features (216 test samples)
    ├── train_transcripts.csv          # Verbatim transcripts (769 train samples)
    ├── test_transcripts.csv           # Verbatim transcripts (216 test samples)
    ├── train_linguistic_features.csv  # Syntactic & lexical metrics (769 train samples)
    ├── test_linguistic_features.csv   # Syntactic & lexical metrics (216 test samples)
    ├── train_judge_scores.csv         # Gemini LLM Judge rubric scores (769 train samples)
    ├── test_judge_scores.csv          # Gemini LLM Judge rubric scores (216 test samples)
    └── evaluation_summary.json        # Serialized evaluation metrics
```

---

## 5. Quickstart & Reproducibility

### 5.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/<YOUR-USERNAME>/shl-spoken-grammar-engine.git
cd shl-spoken-grammar-engine

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5.3 Run Full End-to-End Pipeline
```powershell
python src/run_pipeline.py
```
This executes the feature loading, 5-Fold Stratified Cross-Validation, reports the compulsory training RMSE, and outputs the final `submission.csv`.

---

## 6. Submission Deliverables Checklist

- [x] **`submission.csv`**: Exactly 216 test predictions matching `audio_*.wav` format, continuous in $[0.0, 5.0]$, zero nulls, **Kaggle Leaderboard Score: 0.6077**.
- [x] **`SHL_Grammar_Scoring_Engine.ipynb`**: Complete Jupyter Notebook featuring EDA, feature engineering rationale, multimodal modeling, interpretability scatter plots, and **prominent display of the Training Data RMSE (`0.6367`)**.
- [x] **Public GitHub Repository**: Clean codebase with modular architecture, git-ignored raw data/credentials, and comprehensive documentation.
