# Grammar Scoring Engine: Architecture & Implementation Plan
**Role:** Research Engineer Assessment (SHL)  
**Evaluation Metrics:** Pearson Correlation ($r$) & Root Mean Squared Error ($\text{RMSE}$)  
**Dataset:** 769 Training Audio Samples (16 kHz, mono, ~45–60s) | 216 Test Audio Samples  

---

## 1. Executive Summary & Assessment of Proposed Solution

### Proposed Concept:
> Convert spoken audio into transcripts via Speech-to-Text (ASR), extract grammatical/semantic properties, and use a judge model (G-Eval / LLM-as-a-judge) to score grammar quality against the official rubric.

### Research Engineer Critique & Architectural Strategy:
1. **Transcriptions vs Raw Audio:**
   - Grammar, syntax, clause structures, subject-verb agreement, and tense errors are primarily syntactic/lexical features best captured in the linguistic token domain.
   - **Critical Nuance (ASR Auto-Correction):** Modern ASR models (e.g., Whisper) are trained on massive language corpora and inherently attempt to smooth out or auto-correct spoken grammar slips, false starts, and repetitions unless specifically configured for verbatim transcription (`temperature=0.0`, `condition_on_previous_text=False`).
   - **Acoustic Grounding:** 37 samples in `train.csv` have ground truth `0.0`. These correspond to silent audio, unintelligible recordings, extreme background noise, or unresponsive candidates. A pure text model will fail on these without acoustic energy (RMS) and silence ratio features.

2. **The "Judge Model" (G-Eval / LLM-as-a-Judge):**
   - An LLM judge provides excellent qualitative rubric alignment (understanding nuanced criteria from score 1 to 5).
   - **However, for continuous competition metrics (Pearson Correlation and RMSE):**
     - Raw LLMs output discrete integers (1, 2, 3, 4, 5) with strong central tendency bias.
     - In Kaggle competitions, uncalibrated LLM outputs yield higher RMSE and lower Pearson $r$ compared to supervised regression models trained on the exact ground truth target distribution.
   - **Optimal Architecture:** Use the Judge model to extract structured rubric ratings (Syntax, Morphosyntax, Self-Correction, Overall Rubric MOS). Feed these ratings alongside **Transformer embeddings (DeBERTa-v3/RoBERTa)** and **acoustic features** into a **Calibrated Ensemble Regressor (Ridge / LightGBM)**. This yields the lowest RMSE and highest Pearson correlation possible.

---

## 2. Dataset Diagnostics & Empirical Findings

- **Audio Specs:** 16,000 Hz, 16-bit PCM Mono WAV, duration ~45.0 to 60.5 seconds.
- **Train Split:** 769 audio files in `Dataset_Final/train/`.
- **Test Split:** 216 audio files in `Dataset_Final/test/`.
- **Target MOS Distribution (Train):**
  - Scores: `[0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`
  - Mean: `3.313`, Std: `1.239`, Min: `0.0`, Max: `5.0`
  - Score Breakdown: `0.0` (37), `1.0` (1), `1.5` (3), `2.0` (102), `2.5` (79), `3.0` (174), `3.5` (64), `4.0` (124), `4.5` (52), `5.0` (133).

---

## 3. End-to-End Pipeline Architecture

```
                                  [ Input Spoken Audio (.wav) ]
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        [ Acoustic & Signal Processing ]                   [ Verbatim ASR Transcription (Whisper) ]
        - RMS Amplitude / Energy                           - Verbatim text preserving grammar slips
        - Silence Ratio & Pause Count                      - Word count, token count, duration
        - Speaking Rate (wpm)                              - Disfluency & repetition detection
                       │                                                 │
                       │                               ┌─────────────────┴─────────────────┐
                       │                               ▼                                   ▼
                       │                 [ Linguistic Feature Eng. ]             [ LLM Rubric Judge ]
                       │                 - Syntactic complexity metrics          - Structured Rubric Subscores
                       │                 - Error density indicators              - Zero-shot / Few-shot rubric MOS
                       │                 - POS distribution & diversity                     │
                       │                               │                                    │
                       └───────────────────────┬───────┴────────────────────────────────────┘
                                               ▼
                                  [ Unified Feature Space ]
                                               │
                                               ▼
                       [ 5-Fold Stratified Cross-Validation Regressor ]
                       - Binned Stratified K-Fold (Prevents Data Leakage)
                       - Calibrated Ridge Regressor + LightGBM Regressor
                                               │
                                               ▼
                                 [ Competition Metrics & Output ]
                                 - Leaderboard: Pearson Correlation (r), RMSE
                                 - Compulsory: Display Training Data RMSE
                                 - Final Submission: continuous [0.0, 5.0]
```

---

## 4. Implementation Steps & Execution Plan

### Step 1: Acoustic & Signal Preprocessing
- Extract physical acoustic features for all 769 train and 216 test files:
  - RMS Energy, Peak Amplitude, Mean Energy.
  - Silence Duration, Silence Ratio (below -40dB threshold).
  - Flags for zero-signal / unintelligible audio (predicting the 0.0 class).

### Step 2: Speech-to-Text Transcription (Whisper)
- Transcribe train and test audios using Whisper with verbatim preservation.
- Record transcript length, word count, character count, and speech rate (words per second).
- Save transcripts to persistent Parquet / CSV files so they are cached and inspectable.

### Step 3: Linguistic & LLM Rubric Feature Extraction
- Extract grammar and syntax indicators:
  - Lexical diversity (Type-Token Ratio, lexical density).
  - Sentence structure statistics (mean length, commas/clauses, capitalization, fragmentation).
  - POS tagging frequencies (verbs, adjectives, subordinating conjunctions).
- Extract LLM rubric evaluator scores based on the competition rubric criteria (Scores 1 to 5).

### Step 4: 5-Fold Stratified Cross-Validation & Regression Modeling
- Form 5 stratified folds based on target MOS score brackets.
- Train multi-modal regressor models:
  - Model A: Regularized Ridge Regressor (smooth linear calibration).
  - Model B: LightGBM Regressor (non-linear interaction between acoustics, text features, and rubric scores).
  - Ensemble: Blended predictions minimizing out-of-fold RMSE.
- Compute task-relevant metrics:
  - Pearson Correlation Coefficient ($r$)
  - Root Mean Squared Error ($\text{RMSE}$)
  - Spearman Rank Correlation ($\rho$)
  - Mean Absolute Error ($\text{MAE}$)
  - **Compulsory Requirement: Report Training Data RMSE.**

### Step 5: Test Inference & Submission Generation
- Generate out-of-fold predictions on test set (216 samples).
- Ensure valid schema (`filename,label`), continuous values bounded in $[0.0, 5.0]$.
- Perform sanity checks and generate submission summary.

### Step 6: Publication-Grade Jupyter Notebook Deliverable
- Create a clean, reproducible Jupyter Notebook (`SHL_Grammar_Scoring_Engine.ipynb`) with:
  - Full source code and markdown documentation.
  - Comprehensive report covering methodology, preprocessing, and architecture.
  - Interactive plots: Target distribution, Predicted vs. Actual scatter plot with Pearson regression line, Residual error plot, and Feature importance breakdown.
  - Explicit prominent display of **Training Data RMSE**.
