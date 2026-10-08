import os
import sys
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from scipy.stats import pearsonr

print("=== Training V8 Syntactic Grammar & Speech-Domain Expert ===")

# 1. Load Data
train_df = pd.read_csv("shl-hiring-assessment-2026/Dataset_Final/train.csv")
train_trans = pd.read_csv("data/train_transcripts.csv")
test_trans = pd.read_csv("data/test_transcripts.csv")
train_audio = pd.read_csv("data/train_audio_features.csv")
test_audio = pd.read_csv("data/test_audio_features.csv")
train_ling = pd.read_csv("data/train_linguistic_features.csv")
test_ling = pd.read_csv("data/test_linguistic_features.csv")

if "label" in train_trans.columns:
    train_trans = train_trans.drop(columns=["label"])
if "label" in test_trans.columns:
    test_trans = test_trans.drop(columns=["label"])

# 2. Extract POS Syntactic Tag Sequences for Grammar Error Detection
print("Extracting Part-of-Speech (POS) Syntactic Sequences...")
def extract_pos_sequences(df):
    seqs = []
    for t in df["transcript"].fillna("").astype(str):
        if not t.strip():
            seqs.append("SILENCE")
            continue
        tokens = word_tokenize(t)
        tags = [tag for _, tag in pos_tag(tokens)]
        seqs.append(" ".join(tags))
    return seqs

pos_tr = extract_pos_sequences(train_trans)
pos_te = extract_pos_sequences(test_trans)

# 3. Readability & Lexical Sophistication (TextStat)
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

# 4. Multi-Gram Linguistic Manifolds:
# a) Word N-Grams (1, 2)
# b) Character N-Grams (3, 5)
# c) POS Syntactic N-Grams (2, 4)
print("Extracting Word, Character, and POS N-Grams...")
tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, min_df=2, sublinear_tf=True)
tfidf_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=1500, min_df=3, sublinear_tf=True)
tfidf_pos = TfidfVectorizer(ngram_range=(2, 4), max_features=1200, min_df=3, sublinear_tf=True)

all_tr_text = train_trans["transcript"].fillna("").values
all_te_text = test_trans["transcript"].fillna("").values

X_w_tr = tfidf_word.fit_transform(all_tr_text).toarray()
X_w_te = tfidf_word.transform(all_te_text).toarray()

X_c_tr = tfidf_char.fit_transform(all_tr_text).toarray()
X_c_te = tfidf_char.transform(all_te_text).toarray()

X_p_tr = tfidf_pos.fit_transform(pos_tr).toarray()
X_p_te = tfidf_pos.transform(pos_te).toarray()

X_text_tr = np.hstack([X_w_tr, X_c_tr, X_p_tr])
X_text_te = np.hstack([X_w_te, X_c_te, X_p_te])
print(f"Combined N-Gram text matrix shape: {X_text_tr.shape}")

# SVD Manifold (Top 25 Latent Syntactic Components)
svd = TruncatedSVD(n_components=25, random_state=42)
svd_tr = svd.fit_transform(X_text_tr)
svd_te = svd.transform(X_text_te)

svd_cols = [f"svd_syn_{i}" for i in range(25)]
svd_tr_df = pd.DataFrame(svd_tr, columns=svd_cols)
svd_te_df = pd.DataFrame(svd_te, columns=svd_cols)
svd_tr_df["filename"] = train_trans["filename"]
svd_te_df["filename"] = test_trans["filename"]

# Out-of-fold Ridge on Unified Syntactic N-Grams
y_all = train_df["label"].values
skf_all = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins_all = pd.qcut(y_all, q=5, labels=False, duplicates="drop")

oof_text_ridge = np.zeros(len(y_all))
test_text_ridge = np.zeros(len(test_trans))

for tr, val in skf_all.split(X_text_tr, target_bins_all):
    m = RidgeCV(alphas=np.logspace(-2, 3, 50))
    m.fit(X_text_tr[tr], y_all[tr])
    oof_text_ridge[val] = m.predict(X_text_tr[val])
    test_text_ridge += m.predict(X_text_te) / 5.0

train_trans["syntax_ngram_score"] = oof_text_ridge
test_trans["syntax_ngram_score"] = test_text_ridge

# 5. Clean Acoustic and Linguistic Features (Discard mismatched LLM judge)
ling_cols = [
    "guiraud_index", "num_unique_words", "num_words", "speech_rate",
    "words_per_minute", "mean_word_length", "complex_modal_count", "ttr",
    "num_sentences", "polysyllable_ratio", "relative_pronoun_count",
    "subordination_count", "filler_ratio", "repetition_ratio", "incomplete_sentence_ratio"
]

audio_cols = [
    "duration_sec", "speech_duration_sec", "rms_energy", "speech_ratio",
    "silence_ratio", "zero_crossing_rate", "max_amplitude", "energy_std"
]

train_full = train_audio[["filename"] + audio_cols].merge(
    train_ling[["filename"] + ling_cols], on="filename"
).merge(
    train_trans[["filename", "syntax_ngram_score"]], on="filename"
).merge(
    read_tr, on="filename"
).merge(
    svd_tr_df, on="filename"
)

test_full = test_audio[["filename"] + audio_cols].merge(
    test_ling[["filename"] + ling_cols], on="filename", how="left"
).merge(
    test_trans[["filename", "syntax_ngram_score"]], on="filename", how="left"
).merge(
    read_te, on="filename", how="left"
).merge(
    svd_te_df, on="filename", how="left"
)

# Interaction & Syntactic Complexity Indicators
def add_engineered_features(df):
    df = df.copy()
    df["fluency_pace"] = df["words_per_minute"] / (1.0 + df["speech_duration_sec"])
    df["lexical_sophistication"] = df["guiraud_index"] * df["mean_word_length"]
    df["acoustic_dynamic"] = df["max_amplitude"] / (df["rms_energy"] + 1e-5)
    df["syntactic_depth"] = (df["fk_grade"] + df["coleman_liau"] + df["ari"]) / 3.0
    df["syntax_x_lexical"] = df["syntax_ngram_score"] * df["guiraud_index"]
    return df

train_full = add_engineered_features(train_full)
test_full = add_engineered_features(test_full)

FEATURE_COLS = [c for c in train_full.columns if c not in ["filename", "label", "is_silent_or_corrupt"]]
print(f"Total Features: {len(FEATURE_COLS)}")

# 6. Filter Out Silent Artifacts -> Train Regressors on Pure Speech Domain [1.0, 5.0]
# 100% of test samples are active speech. Training on speech domain aligns target distribution
speech_mask = (train_df["label"] > 0)
X_sp = train_full.loc[speech_mask, FEATURE_COLS].values
y_sp = train_df.loc[speech_mask, "label"].values
X_test = test_full[FEATURE_COLS].values

print(f"Speech Domain Training Set: {X_sp.shape}, Target Mean: {y_sp.mean():.4f}, Std: {y_sp.std():.4f}")

skf_sp = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins_sp = pd.qcut(y_sp, q=5, labels=False, duplicates="drop")

oof_xgb = np.zeros(len(y_sp))
test_xgb = np.zeros(len(test_full))

oof_lgb = np.zeros(len(y_sp))
test_lgb = np.zeros(len(test_full))

oof_cat = np.zeros(len(y_sp))
test_cat = np.zeros(len(test_full))

oof_bridge = np.zeros(len(y_sp))
test_bridge = np.zeros(len(test_full))

for tr, val in skf_sp.split(X_sp, target_bins_sp):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_sp[tr])
    X_va_sc = scaler.transform(X_sp[val])
    X_te_sc = scaler.transform(X_test)

    # 1. XGBoost
    xgb_m = XGBRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.5,
        random_state=42
    )
    xgb_m.fit(X_sp[tr], y_sp[tr])
    oof_xgb[val] = xgb_m.predict(X_sp[val])
    test_xgb += xgb_m.predict(X_test) / 5.0

    # 2. LightGBM
    lgb_m = LGBMRegressor(
        n_estimators=190,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=11,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1
    )
    lgb_m.fit(X_sp[tr], y_sp[tr])
    oof_lgb[val] = lgb_m.predict(X_sp[val])
    test_lgb += lgb_m.predict(X_test) / 5.0

    # 3. CatBoost
    cat_m = CatBoostRegressor(
        iterations=200,
        learning_rate=0.03,
        depth=3,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0
    )
    cat_m.fit(X_sp[tr], y_sp[tr])
    oof_cat[val] = cat_m.predict(X_sp[val])
    test_cat += cat_m.predict(X_test) / 5.0

    # 4. Bayesian Ridge (Regularized Smooth Shrinkage)
    br_m = BayesianRidge()
    br_m.fit(X_tr_sc, y_sp[tr])
    oof_bridge[val] = br_m.predict(X_va_sc)
    test_bridge += br_m.predict(X_te_sc) / 5.0

def eval_m(preds, name):
    r, _ = pearsonr(y_sp, preds)
    rmse = np.sqrt(np.mean((y_sp - preds)**2))
    print(f"[{name:25s}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f}")
    return rmse

print("\n--- Speech-Domain 5-Fold Validation ---")
eval_m(oof_xgb, "XGBoost")
eval_m(oof_lgb, "LightGBM")
eval_m(oof_cat, "CatBoost")
eval_m(oof_bridge, "Bayesian Ridge")

# Solve optimal ensemble weights
def loss(w):
    w1, w2, w3, w4 = w
    pred = np.clip(w1*oof_xgb + w2*oof_lgb + w3*oof_cat + w4*oof_bridge, 1.0, 5.0)
    return np.sqrt(np.mean((y_sp - pred)**2))

res = minimize(loss, [0.4, 0.2, 0.2, 0.2], bounds=[(0, 1)]*4, constraints={"type":"eq", "fun": lambda w: sum(w)-1.0})
opt_w = res.x
print(f"\nOptimal Ensemble Weights: XGB={opt_w[0]:.3f}, LGBM={opt_w[1]:.3f}, CatBoost={opt_w[2]:.3f}, BayesRidge={opt_w[3]:.3f}")

oof_final = np.clip(opt_w[0]*oof_xgb + opt_w[1]*oof_lgb + opt_w[2]*oof_cat + opt_w[3]*oof_bridge, 1.0, 5.0)
final_rmse = eval_m(oof_final, "V8 SYNTACTIC ENSEMBLE")

test_final = np.clip(opt_w[0]*test_xgb + opt_w[1]*test_lgb + opt_w[2]*test_cat + opt_w[3]*test_bridge, 1.0, 5.0)

print(f"\nGround Truth Speech Mean: {y_sp.mean():.4f} | Test Prediction Mean: {test_final.mean():.4f}")
print(f"Ground Truth Speech Std:  {y_sp.std():.4f} | Test Prediction Std:  {test_final.std():.4f}")

# Save submission
sub = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_final, 4)
})
sub["_id"] = sub["filename"].str.extract(r"(\d+)").astype(int)
sub = sub.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)

out_path = "submission_v8_syntactic_expert.csv"
sub.to_csv(out_path, index=False)
sub.to_csv("submission.csv", index=False)
print(f"\n[SUCCESS] Saved {out_path} & submission.csv with {len(sub)} rows!")
print("Sample predictions:\n", sub.head(10))
print("Summary stats:\n", sub["label"].describe())
