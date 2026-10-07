import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor

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

print(f"Loaded Unified Datasets: Train={train_full.shape}, Test={test_full.shape}")

# Feature Engineering: Add high-signal interaction features
def add_features(df):
    df = df.copy()
    # Synergy between LLM judge and acoustics
    df["judge_x_speech_ratio"] = df["judge_rubric_mos"] * df["speech_ratio"]
    df["judge_x_ttr"] = df["judge_rubric_mos"] * df["ttr"]
    df["judge_x_subordination"] = df["judge_rubric_mos"] * df["subordination_ratio"]
    # Fluency interaction
    df["fluency_index"] = df["words_per_minute"] / (1.0 + df["filler_count"] + df["repetition_count"])
    # Acoustic dynamic range
    df["dynamic_energy"] = df["max_amplitude"] / (df["rms_energy"] + 1e-5)
    return df

train_full = add_features(train_full)
test_full = add_features(test_full)

FEATURE_COLS = [
    c for c in train_full.columns 
    if c not in ["filename", "label", "is_silent_or_corrupt", "reasoning"]
]

X = train_full[FEATURE_COLS].values
y = train_full["label"].values
X_test = test_full[FEATURE_COLS].values

# Stratification bins
target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Collect OOF predictions for diverse regressors:
# Model 1: Ridge Regressor
# Model 2: LightGBM Regressor
# Model 3: GradientBoosting Regressor
# Model 4: ExtraTrees Regressor

oof_ridge = np.zeros(len(y))
test_ridge = np.zeros(len(test_full))

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(test_full))

oof_gbr = np.zeros(len(y))
test_gbr = np.zeros(len(test_full))

oof_et = np.zeros(len(y))
test_et = np.zeros(len(test_full))

print("\n--- Training 5-Fold Diverse Models ---")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, target_bins)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_val = X[val_idx], y[val_idx]

    # Standardize
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    # 1. Ridge
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
    ridge.fit(X_tr_sc, y_tr)
    oof_ridge[val_idx] = ridge.predict(X_va_sc)
    test_ridge += ridge.predict(X_te_sc) / n_splits

    # 2. LightGBM
    lgb = LGBMRegressor(
        n_estimators=160,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
    )
    lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb.predict(X_va)
    test_lgb += lgb.predict(X_test) / n_splits

    # 3. GradientBoosting
    gbr = GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.85,
        random_state=42 + fold,
    )
    gbr.fit(X_tr, y_tr)
    oof_gbr[val_idx] = gbr.predict(X_va)
    test_gbr += gbr.predict(X_test) / n_splits

    # 4. ExtraTrees
    et = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et.fit(X_tr, y_tr)
    oof_et[val_idx] = et.predict(X_va)
    test_et += et.predict(X_test) / n_splits

def eval_preds(preds, name):
    r, _ = pearsonr(y, preds)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    mae = np.mean(np.abs(y - preds))
    print(f"[{name}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return r, rmse

print("\n--- Individual Model OOF Scores ---")
eval_preds(oof_ridge, "Ridge Regressor")
eval_preds(oof_lgb, "LightGBM Regressor")
eval_preds(oof_gbr, "GradientBoosting")
eval_preds(oof_et, "ExtraTrees Regressor")

# Acoustic hurdle / silence gating on OOF and Test
silent_train = (train_full["is_silent_or_corrupt"] == 1).values | (train_full["rms_energy"] < 0.003).values
silent_test = (test_full["is_silent_or_corrupt"] == 1).values | (test_full["rms_energy"] < 0.003).values

# Optimize Ensemble Weights directly for RMSE
def loss_func(weights):
    w1, w2, w3, w4 = weights
    pred = w1 * oof_ridge + w2 * oof_lgb + w3 * oof_gbr + w4 * oof_et
    # Apply acoustic gate: if silent/corrupt, label is 0.0
    pred[silent_train] = 0.0
    pred = np.clip(pred, 0.0, 5.0)
    return np.sqrt(np.mean((y - pred) ** 2))

init_weights = [0.2, 0.5, 0.2, 0.1]
bnds = [(0.0, 1.0) for _ in range(4)]
cons = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

res = minimize(loss_func, init_weights, method="SLSQP", bounds=bnds, constraints=cons)
opt_w = res.x
print(f"\n--- Optimized Ensemble Weights ---")
print(f"Ridge: {opt_w[0]:.3f}, LGBM: {opt_w[1]:.3f}, GBR: {opt_w[2]:.3f}, ExtraTrees: {opt_w[3]:.3f}")

oof_blend = opt_w[0] * oof_ridge + opt_w[1] * oof_lgb + opt_w[2] * oof_gbr + opt_w[3] * oof_et
oof_blend[silent_train] = 0.0
oof_blend = np.clip(oof_blend, 0.0, 5.0)
eval_preds(oof_blend, "Optimized Multi-Model Blend (with Acoustic Gate)")

# Variance calibration / scaling factor search
print("\n--- Variance Calibration Optimization ---")
y_mean = np.mean(y[~silent_train])
best_gamma = 1.0
best_calib_rmse = float("inf")

for gamma in np.linspace(0.9, 1.4, 101):
    calib = y_mean + gamma * (oof_blend - y_mean)
    calib[silent_train] = 0.0
    calib = np.clip(calib, 0.0, 5.0)
    rmse = np.sqrt(np.mean((y - calib) ** 2))
    if rmse < best_calib_rmse:
        best_calib_rmse = rmse
        best_gamma = gamma

print(f"Optimal Variance Scaling Factor gamma: {best_gamma:.3f}")
oof_calib = y_mean + best_gamma * (oof_blend - y_mean)
oof_calib[silent_train] = 0.0
oof_calib = np.clip(oof_calib, 0.0, 5.0)
eval_preds(oof_calib, f"Final Calibrated Ensemble (gamma={best_gamma:.3f})")

# Generate test predictions
test_blend = opt_w[0] * test_ridge + opt_w[1] * test_lgb + opt_w[2] * test_gbr + opt_w[3] * test_et
test_calib = y_mean + best_gamma * (test_blend - y_mean)
test_calib[silent_test] = 0.0
test_calib = np.clip(test_calib, 0.0, 5.0)

sub = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_calib, 4)
})
sub["_id"] = sub["filename"].str.extract(r"(\d+)").astype(int)
sub = sub.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)

sub.to_csv("submission_optimized.csv", index=False)
print(f"\n[OK] Saved submission_optimized.csv with {len(sub)} rows!")
print("Sample predictions:\n", sub.head())
