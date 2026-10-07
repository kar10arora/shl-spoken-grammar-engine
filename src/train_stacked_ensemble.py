import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV, BayesianRidge, LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

print("Packages loaded successfully:")
print("LightGBM:", lgb.__version__)
print("XGBoost:", xgb.__version__)
print("CatBoost:", cb.__version__)

# 1. Load data
train_audio = pd.read_csv("data/train_audio_features.csv")
test_audio = pd.read_csv("data/test_audio_features.csv")
train_ling = pd.read_csv("data/train_linguistic_features.csv")
test_ling = pd.read_csv("data/test_linguistic_features.csv")
train_judge = pd.read_csv("data/train_judge_scores.csv")
test_judge = pd.read_csv("data/test_judge_scores.csv")

ling_cols = [c for c in train_ling.columns if c not in ["label", "duration_sec", "speech_duration_sec"]]
judge_cols = ["filename", "judge_rubric_mos", "grammar_accuracy_score", "syntax_score", "sentence_completion_score"]

train_full = train_audio.merge(train_ling[ling_cols], on="filename", how="inner").merge(train_judge[judge_cols], on="filename", how="inner")
test_full = test_audio.merge(test_ling[ling_cols], on="filename", how="left").merge(test_judge[judge_cols], on="filename", how="left")
for col in judge_cols[1:]:
    test_full[col] = test_full[col].fillna(3.0)

print(f"Data shapes -> Train: {train_full.shape}, Test: {test_full.shape}")

# 2. Outlier Handling & Transformation Pipeline
skewed_cols = [
    "filler_count", "repetition_count", "subordination_count", 
    "num_words", "speech_duration_sec", "relative_pronoun_count",
    "complex_modal_count", "self_correction_count"
]

def preprocess_and_engineer(df, is_train=True, train_quantiles=None):
    df = df.copy()
    
    # Interaction & domain features
    df["judge_x_speech_ratio"] = df["judge_rubric_mos"] * df["speech_ratio"]
    df["judge_x_ttr"] = df["judge_rubric_mos"] * df["ttr"]
    df["judge_x_subordination"] = df["judge_rubric_mos"] * df["subordination_ratio"]
    df["fluency_index"] = df["words_per_minute"] / (1.0 + df["filler_count"] + df["repetition_count"])
    df["dynamic_energy"] = df["max_amplitude"] / (df["rms_energy"] + 1e-5)
    
    # Log1p transforms for highly skewed count variables
    for c in skewed_cols:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(np.maximum(df[c], 0))
            
    # Outlier clipping using 1st and 99th percentiles (Winsorization)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ["label", "is_silent_or_corrupt"]]
    quantiles = {}
    if is_train:
        for c in num_cols:
            q_low = df[c].quantile(0.01)
            q_high = df[c].quantile(0.99)
            quantiles[c] = (q_low, q_high)
            df[c] = df[c].clip(q_low, q_high)
        return df, quantiles
    else:
        for c in num_cols:
            if c in train_quantiles:
                q_low, q_high = train_quantiles[c]
                df[c] = df[c].clip(q_low, q_high)
        return df

train_df, quantiles = preprocess_and_engineer(train_full, is_train=True)
test_df = preprocess_and_engineer(test_full, is_train=False, train_quantiles=quantiles)

FEATURE_COLS = [
    c for c in train_df.columns 
    if c not in ["filename", "label", "is_silent_or_corrupt", "reasoning"]
]

print(f"Engineered Feature Count: {len(FEATURE_COLS)}")

X = train_df[FEATURE_COLS].values
y = train_df["label"].values
X_test = test_df[FEATURE_COLS].values

# Stratified 5-Fold split based on target quintiles
target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Level-0 Base Models:
# 1. LightGBM
# 2. CatBoost
# 3. XGBoost
# 4. GradientBoosting (Scikit-Learn)
# 5. Regularized Ridge (Linear)

oof_lgb = np.zeros(len(y))
oof_cat = np.zeros(len(y))
oof_xgb = np.zeros(len(y))
oof_gbr = np.zeros(len(y))
oof_ridge = np.zeros(len(y))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_gbr = np.zeros(len(test_df))
test_ridge = np.zeros(len(test_df))

print("\n--- Training Level-0 Diverse Models (5-Fold Stratified) ---")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, target_bins)):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_val = X[val_idx], y[val_idx]

    # Robust scaling for linear model
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    # 1. LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=14,
        min_child_samples=16,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / n_splits

    # 2. CatBoost
    model_cat = cb.CatBoostRegressor(
        iterations=200,
        learning_rate=0.04,
        depth=4,
        l2_leaf_reg=4.0,
        random_seed=42 + fold,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test) / n_splits

    # 3. XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=160,
        learning_rate=0.035,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42 + fold
    )
    model_xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / n_splits

    # 4. GradientBoosting
    model_gbr = GradientBoostingRegressor(
        n_estimators=140,
        learning_rate=0.035,
        max_depth=3,
        subsample=0.85,
        random_state=42 + fold
    )
    model_gbr.fit(X_tr, y_tr)
    oof_gbr[val_idx] = model_gbr.predict(X_va)
    test_gbr += model_gbr.predict(X_test) / n_splits

    # 5. Ridge
    model_ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
    model_ridge.fit(X_tr_sc, y_tr)
    oof_ridge[val_idx] = model_ridge.predict(X_va_sc)
    test_ridge += model_ridge.predict(X_te_sc) / n_splits

def print_metrics(y_true, y_pred, name):
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"[{name}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse

print("\n--- Level-0 Model Cross-Validation Metrics ---")
print_metrics(y, oof_lgb, "1. LightGBM")
print_metrics(y, oof_cat, "2. CatBoost")
print_metrics(y, oof_xgb, "3. XGBoost")
print_metrics(y, oof_gbr, "4. GradientBoosting")
print_metrics(y, oof_ridge, "5. Ridge (Robust Scaled)")

# --- Level-1 Meta-Model: Stacking Generalization ---
# Matrix of OOF predictions
OOF_META = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_gbr, oof_ridge])
TEST_META = np.column_stack([test_lgb, test_cat, test_xgb, test_gbr, test_ridge])

# 1. Non-Negative Least Squares / Constrained Meta-Weights minimizing RMSE directly
def meta_loss(w):
    pred = np.dot(OOF_META, w)
    pred = np.clip(pred, 0.0, 5.0)
    return np.sqrt(np.mean((y - pred) ** 2))

init_w = np.ones(5) / 5.0
bounds = [(0.0, 1.0) for _ in range(5)]
constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(meta_loss, init_w, method="SLSQP", bounds=bounds, constraints=constraint)
opt_weights = opt_res.x

print("\n--- Level-1 Stacking Meta-Weights ---")
model_names = ["LightGBM", "CatBoost", "XGBoost", "GradientBoosting", "Ridge"]
for name, w in zip(model_names, opt_weights):
    print(f"  * {name:18s}: {w*100:5.1f}%")

# Generate Stacking OOF predictions
oof_stacked = np.dot(OOF_META, opt_weights)
oof_stacked = np.clip(oof_stacked, 0.0, 5.0)
print("\n--- Final Stacked Model OOF Performance ---")
final_rmse = print_metrics(y, oof_stacked, "STACKED ENSEMBLE (LGBM+CatBoost+XGB+GBR+Ridge)")

# Generate Test Predictions
test_stacked = np.dot(TEST_META, opt_weights)
test_stacked = np.clip(test_stacked, 0.0, 5.0)

# Build and naturally sort final submission
sub_stacked = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_stacked, 4)
})
sub_stacked["_id"] = sub_stacked["filename"].str.extract(r"(\d+)").astype(int)
sub_stacked = sub_stacked.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)

# Save to submission_stacked.csv AND submission.csv
sub_stacked.to_csv("submission_stacked.csv", index=False)
sub_stacked.to_csv("submission.csv", index=False)
sub_stacked.to_csv("data/submission.csv", index=False)

print(f"\n[OK] Saved submission_stacked.csv & submission.csv with {len(sub_stacked)} rows!")
print("Sample predictions:\n", sub_stacked.head(10))
print("\nPrediction Summary Statistics:")
print(sub_stacked["label"].describe())
