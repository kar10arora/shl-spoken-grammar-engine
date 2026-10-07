import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor

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

# Add interactions
def add_features(df):
    df = df.copy()
    df["judge_x_speech_ratio"] = df["judge_rubric_mos"] * df["speech_ratio"]
    df["judge_x_ttr"] = df["judge_rubric_mos"] * df["ttr"]
    df["judge_x_subordination"] = df["judge_rubric_mos"] * df["subordination_ratio"]
    df["fluency_index"] = df["words_per_minute"] / (1.0 + df["filler_count"] + df["repetition_count"])
    df["dynamic_energy"] = df["max_amplitude"] / (df["rms_energy"] + 1e-5)
    return df

train_full = add_features(train_full)
test_full = add_features(test_full)

FEATURE_COLS = [c for c in train_full.columns if c not in ["filename", "label", "is_silent_or_corrupt", "reasoning"]]

# Filter out silent/corrupt for speech regression
speech_mask = (train_full["is_silent_or_corrupt"] == 0) & (train_full["rms_energy"] >= 0.003) & (train_full["label"] > 0)
train_speech = train_full[speech_mask].copy().reset_index(drop=True)

X_sp = train_speech[FEATURE_COLS].values
y_sp = train_speech["label"].values
X_test = test_full[FEATURE_COLS].values

print(f"Speech-only training set: {X_sp.shape}, mean label: {y_sp.mean():.4f}, std: {y_sp.std():.4f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins = pd.qcut(y_sp, q=5, labels=False, duplicates="drop")

oof_ridge = np.zeros(len(y_sp))
oof_lgb = np.zeros(len(y_sp))
oof_gbr = np.zeros(len(y_sp))
oof_et = np.zeros(len(y_sp))

test_ridge = np.zeros(len(test_full))
test_lgb = np.zeros(len(test_full))
test_gbr = np.zeros(len(test_full))
test_et = np.zeros(len(test_full))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sp, target_bins)):
    X_tr, y_tr = X_sp[tr_idx], y_sp[tr_idx]
    X_va, y_val = X_sp[val_idx], y_sp[val_idx]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
    ridge.fit(X_tr_sc, y_tr)
    oof_ridge[val_idx] = ridge.predict(X_va_sc)
    test_ridge += ridge.predict(X_te_sc) / 5

    lgb = LGBMRegressor(n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12, min_child_samples=15, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1)
    lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb.predict(X_va)
    test_lgb += lgb.predict(X_test) / 5

    gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.04, max_depth=3, subsample=0.85, random_state=42+fold)
    gbr.fit(X_tr, y_tr)
    oof_gbr[val_idx] = gbr.predict(X_va)
    test_gbr += gbr.predict(X_test) / 5

    et = ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_split=8, random_state=42+fold, n_jobs=2)
    et.fit(X_tr, y_tr)
    oof_et[val_idx] = et.predict(X_va)
    test_et += et.predict(X_test) / 5

def eval_m(y_true, y_pred, name):
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"[{name}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse

print("\n--- Speech-only Model Performance ---")
eval_m(y_sp, oof_ridge, "Ridge (Speech Only)")
eval_m(y_sp, oof_lgb, "LightGBM (Speech Only)")
eval_m(y_sp, oof_gbr, "GBR (Speech Only)")
eval_m(y_sp, oof_et, "ExtraTrees (Speech Only)")

# Blend
def loss_f(w):
    w1, w2, w3, w4 = w
    pred = np.clip(w1*oof_ridge + w2*oof_lgb + w3*oof_gbr + w4*oof_et, 1.0, 5.0)
    return np.sqrt(np.mean((y_sp - pred)**2))

res = minimize(loss_f, [0.2, 0.4, 0.3, 0.1], bounds=[(0,1)]*4, constraints={"type":"eq", "fun": lambda w: sum(w)-1})
opt_w = res.x
print(f"Optimal weights: Ridge={opt_w[0]:.3f}, LGBM={opt_w[1]:.3f}, GBR={opt_w[2]:.3f}, ET={opt_w[3]:.3f}")

oof_blend = np.clip(opt_w[0]*oof_ridge + opt_w[1]*oof_lgb + opt_w[2]*oof_gbr + opt_w[3]*oof_et, 1.0, 5.0)
eval_m(y_sp, oof_blend, "Optimized Speech Blend")

# Variance calibration
y_mean = np.mean(y_sp)
best_gamma = 1.0
best_rmse = float("inf")
for g in np.linspace(0.9, 1.4, 101):
    c = np.clip(y_mean + g*(oof_blend - y_mean), 1.0, 5.0)
    rmse = np.sqrt(np.mean((y_sp - c)**2))
    if rmse < best_rmse:
        best_rmse = rmse
        best_gamma = g

oof_calib = np.clip(y_mean + best_gamma*(oof_blend - y_mean), 1.0, 5.0)
print(f"Best gamma: {best_gamma:.3f}")
eval_m(y_sp, oof_calib, "Calibrated Speech Blend")

# Now check overall evaluation including the 37 silent files (predict 0.0 on silent files)
all_oof_preds = np.zeros(len(train_full))
all_oof_preds[speech_mask] = oof_calib
all_oof_preds[~speech_mask] = 0.0

r_all, _ = pearsonr(train_full["label"], all_oof_preds)
rmse_all = np.sqrt(np.mean((train_full["label"] - all_oof_preds)**2))
print(f"\n=======================================================")
print(f"OVERALL DATASET (Including 37 silent files correctly gated to 0.0):")
print(f"  -> Pearson r: {r_all:.4f}")
print(f"  -> RMSE:      {rmse_all:.4f}")
print(f"=======================================================")
