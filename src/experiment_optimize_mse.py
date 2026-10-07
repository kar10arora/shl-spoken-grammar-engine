import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr

# 1. Load data
train_audio = pd.read_csv("data/train_audio_features.csv")
test_audio = pd.read_csv("data/test_audio_features.csv")
train_ling = pd.read_csv("data/train_linguistic_features.csv")
test_ling = pd.read_csv("data/test_linguistic_features.csv")
train_judge = pd.read_csv("data/train_judge_scores.csv")
test_judge = pd.read_csv("data/test_judge_scores.csv")

ling_cols = [c for c in train_ling.columns if c not in ["label", "duration_sec", "speech_duration_sec"]]
judge_cols = ["filename", "judge_rubric_mos", "grammar_accuracy_score", "syntax_score", "sentence_completion_score"]

train_full = train_audio.merge(train_ling[ling_cols], on="filename", how="inner")
train_full = train_full.merge(train_judge[judge_cols], on="filename", how="inner")

test_full = test_audio.merge(test_ling[ling_cols], on="filename", how="left")
test_full = test_full.merge(test_judge[judge_cols], on="filename", how="left")
for col in judge_cols[1:]:
    test_full[col] = test_full[col].fillna(3.0)

print(f"Train full: {train_full.shape}, Test full: {test_full.shape}")

# Feature columns
FEATURE_COLS = [c for c in train_full.columns if c not in ["filename", "label", "is_silent_or_corrupt"]]

y = train_full["label"].values
silent_mask = (train_full["is_silent_or_corrupt"] == 1).values | (train_full["rms_energy"] < 0.003).values

# Check target stats
print(f"Target stats: Mean={y.mean():.4f}, Std={y.std():.4f}, Min={y.min()}, Max={y.max()}")
print(f"Number of silent/corrupt train samples: {silent_mask.sum()}")
print(f"Non-silent target stats: Mean={y[~silent_mask].mean():.4f}, Std={y[~silent_mask].std():.4f}")

# Correlation of features with label
corrs = train_full[FEATURE_COLS].apply(lambda col: pearsonr(col.fillna(col.median()), y)[0])
print("\nTop 10 correlated features with ground truth label:")
print(corrs.abs().sort_values(ascending=False).head(10))
