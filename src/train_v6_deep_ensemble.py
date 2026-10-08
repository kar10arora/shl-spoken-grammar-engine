import os
import sys
import numpy as np
import pandas as pd
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV, Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize
from scipy.stats import pearsonr

print("="*60)
print("V6 Deep Hybrid Engine: Neural Net (MLP) + Bagged XGBoost + LightGBM + BayesRidge")
print("="*60)

# 1. Load Data
train_df = pd.read_csv("shl-hiring-assessment-2026/Dataset_Final/train.csv")
train_trans = pd.read_csv("data/train_transcripts.csv")
test_trans = pd.read_csv("data/test_transcripts.csv")

train_audio = pd.read_csv("data/train_audio_features.csv")
test_audio = pd.read_csv("data/test_audio_features.csv")

train_ling = pd.read_csv("data/train_linguistic_features.csv")
test_ling = pd.read_csv("data/test_linguistic_features.csv")

train_judge = pd.read_csv("data/train_judge_scores.csv")
test_judge = pd.read_csv("data/test_judge_scores.csv")

if "label" in train_trans.columns:
    train_trans = train_trans.drop(columns=["label"])
if "label" in test_trans.columns:
    test_trans = test_trans.drop(columns=["label"])

y = train_df["label"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

# 2. Extract Readability Indices
def extract_readability(df):
    records = []
    for t in df["transcript"].fillna(""):
        if not t.strip():
            records.append({
                "read_ease": 50.0, "fk_grade": 5.0, "coleman_liau": 5.0,
                "ari": 5.0, "dale_chall": 5.0, "fog": 5.0, "diff_words_ratio": 0.0
            })
            continue
        words = t.split()
        n_w = max(len(words), 1)
        records.append({
            "read_ease": textstat.flesch_reading_ease(t),
            "fk_grade": textstat.flesch_kincaid_grade(t),
            "coleman_liau": textstat.coleman_liau_index(t),
            "ari": textstat.automated_readability_index(t),
            "dale_chall": textstat.dale_chall_readability_score(t),
            "fog": textstat.gunning_fog(t),
            "diff_words_ratio": textstat.difficult_words(t) / n_w
        })
    res_df = pd.DataFrame(records)
    res_df["filename"] = df["filename"].values
    return res_df

read_tr = extract_readability(train_trans)
read_te = extract_readability(test_trans)

# 3. Sublinear N-Grams + 20 SVD Manifolds
tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=1800, min_df=2, sublinear_tf=True)
tfidf_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=1800, min_df=3, sublinear_tf=True)

all_tr_text = train_trans["transcript"].fillna("").values
all_te_text = test_trans["transcript"].fillna("").values

X_w_tr = tfidf_word.fit_transform(all_tr_text).toarray()
X_w_te = tfidf_word.transform(all_te_text).toarray()

X_c_tr = tfidf_char.fit_transform(all_tr_text).toarray()
X_c_te = tfidf_char.transform(all_te_text).toarray()

X_text_tr = np.hstack([X_w_tr, X_c_tr])
X_text_te = np.hstack([X_w_te, X_c_te])

oof_text_ridge = np.zeros(len(y))
test_text_ridge = np.zeros(len(test_trans))

for tr, val in skf.split(X_text_tr, target_bins):
    m = RidgeCV(alphas=np.logspace(-2, 3, 50))
    m.fit(X_text_tr[tr], y[tr])
    oof_text_ridge[val] = m.predict(X_text_tr[val])
    test_text_ridge += m.predict(X_text_te) / 5.0

svd = TruncatedSVD(n_components=20, random_state=42)
svd_tr = svd.fit_transform(X_text_tr)
svd_te = svd.transform(X_text_te)

svd_cols = [f"svd_text_{i}" for i in range(20)]
svd_tr_df = pd.DataFrame(svd_tr, columns=svd_cols)
svd_te_df = pd.DataFrame(svd_te, columns=svd_cols)
svd_tr_df["filename"] = train_trans["filename"]
svd_te_df["filename"] = test_trans["filename"]

train_trans["text_ngram_score"] = oof_text_ridge
test_trans["text_ngram_score"] = test_text_ridge

# 4. Merge All Modalities
ling_cols = [c for c in train_ling.columns if c not in ["label", "duration_sec", "speech_duration_sec"]]
judge_cols = ["filename", "judge_rubric_mos", "grammar_accuracy_score", "syntax_score", "sentence_completion_score"]

train_full = train_audio.merge(train_ling[ling_cols], on="filename").merge(train_judge[judge_cols], on="filename").merge(train_trans[["filename", "text_ngram_score"]], on="filename").merge(read_tr, on="filename").merge(svd_tr_df, on="filename")
test_full = test_audio.merge(test_ling[ling_cols], on="filename", how="left").merge(test_judge[judge_cols], on="filename", how="left").merge(test_trans[["filename", "text_ngram_score"]], on="filename", how="left").merge(read_te, on="filename", how="left").merge(svd_te_df, on="filename", how="left")

for col in judge_cols[1:]:
    test_full[col] = test_full[col].fillna(3.0)

def add_features(df):
    df = df.copy()
    df["judge_x_speech_ratio"] = df["judge_rubric_mos"] * df["speech_ratio"]
    df["judge_x_ttr"] = df["judge_rubric_mos"] * df["ttr"]
    df["judge_x_text_ngram"] = df["judge_rubric_mos"] * df["text_ngram_score"]
    df["fluency_index"] = df["words_per_minute"] / (1.0 + df["filler_count"] + df["repetition_count"])
    df["dynamic_energy"] = df["max_amplitude"] / (df["rms_energy"] + 1e-5)
    df["complexity_grade"] = (df["fk_grade"] + df["coleman_liau"] + df["ari"]) / 3.0
    return df

train_full = add_features(train_full)
test_full = add_features(test_full)

FEATURE_COLS = [c for c in train_full.columns if c not in ["filename", "label", "is_silent_or_corrupt"]]
print(f"Total Unified Features: {len(FEATURE_COLS)}")

X = train_full[FEATURE_COLS].values
X_test = test_full[FEATURE_COLS].values

# 5. Multi-Model Heterogeneous Training across 5 Folds:
# A) Bagged XGBoost (Seeds 42, 101, 777)
# B) Bagged LightGBM (Seeds 42, 101, 777)
# C) Multi-Layer Perceptron Neural Network (MLPRegressor)
# D) Bayesian Ridge Regressor

oof_xgb = np.zeros(len(y))
test_xgb = np.zeros(len(test_full))

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(test_full))

oof_mlp = np.zeros(len(y))
test_mlp = np.zeros(len(test_full))

oof_bridge = np.zeros(len(y))
test_bridge = np.zeros(len(test_full))

SEEDS = [42, 101, 777]

for tr, val in skf.split(X, target_bins):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X[tr])
    X_va_sc = scaler.transform(X[val])
    X_te_sc = scaler.transform(X_test)

    # 1. XGBoost
    for s in SEEDS:
        xgb_m = XGBRegressor(
            n_estimators=180,
            learning_rate=0.025,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.4,
            reg_lambda=2.5,
            random_state=s
        )
        xgb_m.fit(X[tr], y[tr])
        oof_xgb[val] += xgb_m.predict(X[val]) / (len(SEEDS))
        test_xgb += xgb_m.predict(X_test) / (5.0 * len(SEEDS))

        # 2. LightGBM
        lgb_m = LGBMRegressor(
            n_estimators=190,
            learning_rate=0.025,
            max_depth=3,
            num_leaves=10,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=2.5,
            random_state=s,
            verbose=-1
        )
        lgb_m.fit(X[tr], y[tr])
        oof_lgb[val] += lgb_m.predict(X[val]) / (len(SEEDS))
        test_lgb += lgb_m.predict(X_test) / (5.0 * len(SEEDS))

    # 3. Neural Network (MLPRegressor: 2 hidden layers 64, 32 with heavy L2 regularization)
    mlp_m = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.1,
        learning_rate_init=0.01,
        max_iter=250,
        early_stopping=True,
        random_state=42
    )
    mlp_m.fit(X_tr_sc, y[tr])
    oof_mlp[val] = mlp_m.predict(X_va_sc)
    test_mlp += mlp_m.predict(X_te_sc) / 5.0

    # 4. Bayesian Ridge
    br_m = BayesianRidge()
    br_m.fit(X_tr_sc, y[tr])
    oof_bridge[val] = br_m.predict(X_va_sc)
    test_bridge += br_m.predict(X_te_sc) / 5.0

def eval_m(preds, name):
    r, _ = pearsonr(y, preds)
    rmse = np.sqrt(np.mean((y - preds)**2))
    print(f"[{name:20s}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f}")
    return rmse

print("\n--- Model Cross-Validation Metrics ---")
eval_m(oof_xgb, "Bagged XGBoost")
eval_m(oof_lgb, "Bagged LightGBM")
eval_m(oof_mlp, "Neural Net (MLP)")
eval_m(oof_bridge, "Bayesian Ridge")

# Optimize 4-way blend weights directly for RMSE
def loss(w):
    w1, w2, w3, w4 = w
    pred = np.clip(w1*oof_xgb + w2*oof_lgb + w3*oof_mlp + w4*oof_bridge, 0.0, 5.0)
    return np.sqrt(np.mean((y - pred)**2))

res = minimize(loss, [0.5, 0.2, 0.1, 0.2], bounds=[(0, 1)]*4, constraints={"type":"eq", "fun": lambda w: sum(w)-1.0})
opt_w = res.x
print(f"\nOptimal Ensemble Weights: XGBoost={opt_w[0]:.3f}, LGBM={opt_w[1]:.3f}, NeuralNet={opt_w[2]:.3f}, BayesRidge={opt_w[3]:.3f}")

oof_final = np.clip(opt_w[0]*oof_xgb + opt_w[1]*oof_lgb + opt_w[2]*oof_mlp + opt_w[3]*oof_bridge, 0.0, 5.0)
final_rmse = eval_m(oof_final, "V6 DEEP HYBRID BLEND")

test_final = np.clip(opt_w[0]*test_xgb + opt_w[1]*test_lgb + opt_w[2]*test_mlp + opt_w[3]*test_bridge, 0.0, 5.0)

sub = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_final, 4)
})
sub["_id"] = sub["filename"].str.extract(r"(\d+)").astype(int)
sub = sub.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)

sub.to_csv("submission_v6_deep_hybrid.csv", index=False)
print(f"\n[OK] Saved submission_v6_deep_hybrid.csv with {len(sub)} rows!")
print("Sample predictions:\n", sub.head())
print("Summary stats:\n", sub["label"].describe())
