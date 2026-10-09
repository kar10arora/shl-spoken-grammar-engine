import os
import sys
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
import textstat
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from scipy.stats import pearsonr

print("=== Training V11 Dual-Transformer Titan Engine (MPNet-768 + MiniLM-384) ===")

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

all_tr_text = train_trans["transcript"].fillna("").astype(str).tolist()
all_te_text = test_trans["transcript"].fillna("").astype(str).tolist()

clean_tr_text = [t.strip() if t.strip() else "[silent audio]" for t in all_tr_text]
clean_te_text = [t.strip() if t.strip() else "[silent audio]" for t in all_te_text]

# 2. Extract Dual Transformer Embeddings:
# Model A: all-mpnet-base-v2 (768 Dimensions - Top Ranked MTEB Syntactic/Semantic Architecture)
# Model B: all-MiniLM-L6-v2 (384 Dimensions - Fast Local Discourse Coherence)
# File paths for cached embeddings
f_mpnet_tr = "data/train_mpnet_embeddings.npy"
f_mpnet_te = "data/test_mpnet_embeddings.npy"
f_minilm_tr = "data/train_minilm_embeddings.npy"
f_minilm_te = "data/test_minilm_embeddings.npy"

if os.path.exists(f_mpnet_tr) and os.path.exists(f_mpnet_te):
    print("1/5 Loading cached all-mpnet-base-v2 (768-D)...")
    embs_mpnet_tr = np.load(f_mpnet_tr)
    embs_mpnet_te = np.load(f_mpnet_te)
else:
    print("1/5 Loading and Encoding with all-mpnet-base-v2 (768-D)...")
    mpnet = SentenceTransformer("all-mpnet-base-v2")
    embs_mpnet_tr = mpnet.encode(clean_tr_text, batch_size=64, show_progress_bar=False)
    embs_mpnet_te = mpnet.encode(clean_te_text, batch_size=64, show_progress_bar=False)
    np.save(f_mpnet_tr, embs_mpnet_tr)
    np.save(f_mpnet_te, embs_mpnet_te)
print(f"--> MPNet Extracted: Train={embs_mpnet_tr.shape}, Test={embs_mpnet_te.shape}")

if os.path.exists(f_minilm_tr) and os.path.exists(f_minilm_te):
    print("2/5 Loading cached all-MiniLM-L6-v2 (384-D)...")
    embs_minilm_tr = np.load(f_minilm_tr)
    embs_minilm_te = np.load(f_minilm_te)
else:
    print("2/5 Loading and Encoding with all-MiniLM-L6-v2 (384-D)...")
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    embs_minilm_tr = minilm.encode(clean_tr_text, batch_size=64, show_progress_bar=False)
    embs_minilm_te = minilm.encode(clean_te_text, batch_size=64, show_progress_bar=False)
    np.save(f_minilm_tr, embs_minilm_tr)
    np.save(f_minilm_te, embs_minilm_te)
print(f"--> MiniLM Extracted: Train={embs_minilm_tr.shape}, Test={embs_minilm_te.shape}")

# Combined 1152-Dimensional Dense Transformer Space
embs_dual_tr = np.hstack([embs_mpnet_tr, embs_minilm_tr])
embs_dual_te = np.hstack([embs_mpnet_te, embs_minilm_te])
print(f"--> Unified 1152-D Dense Transformer Tensor: {embs_dual_tr.shape}")

# 3. Dimensionality Reduction & Latent Semantic Manifolds (30 Components)
svd_trans = TruncatedSVD(n_components=30, random_state=42)
trans_svd_tr = svd_trans.fit_transform(embs_dual_tr)
trans_svd_te = svd_trans.transform(embs_dual_te)

trans_svd_cols = [f"trans_svd_{i}" for i in range(30)]
trans_tr_df = pd.DataFrame(trans_svd_tr, columns=trans_svd_cols)
trans_te_df = pd.DataFrame(trans_svd_te, columns=trans_svd_cols)
trans_tr_df["filename"] = train_trans["filename"]
trans_te_df["filename"] = test_trans["filename"]

# Out-of-fold Dual-Transformer Ridge Regression
y_all = train_df["label"].values
skf_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
target_bins_all = pd.qcut(y_all, q=5, labels=False, duplicates="drop")

oof_trans_ridge = np.zeros(len(y_all))
test_trans_ridge = np.zeros(len(test_trans))

for tr, val in skf_10.split(embs_dual_tr, target_bins_all):
    m = RidgeCV(alphas=np.logspace(-2, 3, 50))
    m.fit(embs_dual_tr[tr], y_all[tr])
    oof_trans_ridge[val] = m.predict(embs_dual_tr[val])
    test_trans_ridge += m.predict(embs_dual_te) / 10.0

train_trans["trans_semantic_score"] = oof_trans_ridge
test_trans["trans_semantic_score"] = test_trans_ridge
print(f"--> Dual Transformer Ridge Alone -> Pearson r: {pearsonr(y_all, oof_trans_ridge)[0]:.4f}")

# 4. Syntactic POS Sequences & TextStat Readability
print("3/5 Extracting POS Sequences and Readability Indices...")
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

# Multi-Gram TF-IDF with SVD (25 Syntactic Components)
tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=1800, min_df=2, sublinear_tf=True)
tfidf_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=1800, min_df=3, sublinear_tf=True)
tfidf_pos = TfidfVectorizer(ngram_range=(2, 4), max_features=1500, min_df=3, sublinear_tf=True)

X_w_tr = tfidf_word.fit_transform(all_tr_text).toarray()
X_w_te = tfidf_word.transform(all_te_text).toarray()

X_c_tr = tfidf_char.fit_transform(all_tr_text).toarray()
X_c_te = tfidf_char.transform(all_te_text).toarray()

X_p_tr = tfidf_pos.fit_transform(pos_tr).toarray()
X_p_te = tfidf_pos.transform(pos_te).toarray()

X_text_tr = np.hstack([X_w_tr, X_c_tr, X_p_tr])
X_text_te = np.hstack([X_w_te, X_c_te, X_p_te])

svd_syn = TruncatedSVD(n_components=25, random_state=42)
svd_tr = svd_syn.fit_transform(X_text_tr)
svd_te = svd_syn.transform(X_text_te)

svd_cols = [f"svd_syn_{i}" for i in range(25)]
svd_tr_df = pd.DataFrame(svd_tr, columns=svd_cols)
svd_te_df = pd.DataFrame(svd_te, columns=svd_cols)
svd_tr_df["filename"] = train_trans["filename"]
svd_te_df["filename"] = test_trans["filename"]

oof_text_ridge = np.zeros(len(y_all))
test_text_ridge = np.zeros(len(test_trans))

for tr, val in skf_10.split(X_text_tr, target_bins_all):
    m = RidgeCV(alphas=np.logspace(-2, 3, 50))
    m.fit(X_text_tr[tr], y_all[tr])
    oof_text_ridge[val] = m.predict(X_text_tr[val])
    test_text_ridge += m.predict(X_text_te) / 10.0

train_trans["syntax_ngram_score"] = oof_text_ridge
test_trans["syntax_ngram_score"] = test_text_ridge

# 5. Clean Acoustic & Linguistic Modalities
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
    train_trans[["filename", "syntax_ngram_score", "trans_semantic_score"]], on="filename"
).merge(
    read_tr, on="filename"
).merge(
    svd_tr_df, on="filename"
).merge(
    trans_tr_df, on="filename"
)

test_full = test_audio[["filename"] + audio_cols].merge(
    test_ling[["filename"] + ling_cols], on="filename", how="left"
).merge(
    test_trans[["filename", "syntax_ngram_score", "trans_semantic_score"]], on="filename", how="left"
).merge(
    read_te, on="filename", how="left"
).merge(
    svd_te_df, on="filename", how="left"
).merge(
    trans_te_df, on="filename", how="left"
)

# Cross-Modal Interaction Features
def add_engineered_features(df):
    df = df.copy()
    df["fluency_pace"] = df["words_per_minute"] / (1.0 + df["speech_duration_sec"])
    df["lexical_sophistication"] = df["guiraud_index"] * df["mean_word_length"]
    df["syntactic_depth"] = (df["fk_grade"] + df["coleman_liau"] + df["ari"]) / 3.0
    df["syntax_x_lexical"] = df["syntax_ngram_score"] * df["guiraud_index"]
    df["semantic_x_syntax"] = df["trans_semantic_score"] * df["syntax_ngram_score"]
    df["semantic_x_lexical"] = df["trans_semantic_score"] * df["guiraud_index"]
    df["semantic_x_depth"] = df["trans_semantic_score"] * df["syntactic_depth"]
    return df

train_full = add_engineered_features(train_full)
test_full = add_engineered_features(test_full)

FEATURE_COLS = [c for c in train_full.columns if c not in ["filename", "label", "is_silent_or_corrupt"]]
print(f"Total Unified Features: {len(FEATURE_COLS)}")

# 6. Train on Speech Domain with 10-Fold Stratified CV
speech_mask = (train_df["label"] > 0)
X_sp = train_full.loc[speech_mask, FEATURE_COLS].values
y_sp = train_df.loc[speech_mask, "label"].values
X_test = test_full[FEATURE_COLS].values

print(f"4/5 Training 10-Fold Regressors on Speech Domain ({X_sp.shape[0]} samples, {X_sp.shape[1]} features)...")

skf_sp = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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
        n_estimators=220,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.5,
        random_state=42
    )
    xgb_m.fit(X_sp[tr], y_sp[tr])
    oof_xgb[val] = xgb_m.predict(X_sp[val])
    test_xgb += xgb_m.predict(X_test) / 10.0

    # 2. LightGBM
    lgb_m = LGBMRegressor(
        n_estimators=220,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=11,
        min_child_samples=18,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1
    )
    lgb_m.fit(X_sp[tr], y_sp[tr])
    oof_lgb[val] = lgb_m.predict(X_sp[val])
    test_lgb += lgb_m.predict(X_test) / 10.0

    # 3. CatBoost
    cat_m = CatBoostRegressor(
        iterations=240,
        learning_rate=0.025,
        depth=3,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0
    )
    cat_m.fit(X_sp[tr], y_sp[tr])
    oof_cat[val] = cat_m.predict(X_sp[val])
    test_cat += cat_m.predict(X_test) / 10.0

    # 4. Bayesian Ridge
    br_m = BayesianRidge()
    br_m.fit(X_tr_sc, y_sp[tr])
    oof_bridge[val] = br_m.predict(X_va_sc)
    test_bridge += br_m.predict(X_te_sc) / 10.0

def eval_m(preds, name):
    r, _ = pearsonr(y_sp, preds)
    rmse = np.sqrt(np.mean((y_sp - preds)**2))
    mse = np.mean((y_sp - preds)**2)
    print(f"[{name:30s}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f} | MSE: {mse:.4f}")
    return rmse

print("\n--- 10-Fold Cross-Validation Scores with Dual-Transformer Tensor ---")
eval_m(oof_xgb, "XGBoost + Dual-Transformer")
eval_m(oof_lgb, "LightGBM + Dual-Transformer")
eval_m(oof_cat, "CatBoost + Dual-Transformer")
eval_m(oof_bridge, "Bayesian Ridge + Dual-Trans")

# Optimize MSE loss directly
def loss(w):
    pred = np.clip(w[0]*oof_xgb + w[1]*oof_lgb + w[2]*oof_cat + w[3]*oof_bridge, 1.0, 5.0)
    return np.mean((y_sp - pred)**2)

res = minimize(loss, [0.35, 0.15, 0.35, 0.15], bounds=[(0, 1)]*4, constraints={"type":"eq", "fun": lambda w: sum(w)-1.0})
opt_w = res.x
print(f"\nOptimal Weights: XGB={opt_w[0]:.3f}, LGBM={opt_w[1]:.3f}, CatBoost={opt_w[2]:.3f}, BayesRidge={opt_w[3]:.3f}")

oof_v11 = np.clip(opt_w[0]*oof_xgb + opt_w[1]*oof_lgb + opt_w[2]*oof_cat + opt_w[3]*oof_bridge, 1.0, 5.0)
eval_m(oof_v11, "V11 DUAL-TRANSFORMER TITAN")

test_v11 = np.clip(opt_w[0]*test_xgb + opt_w[1]*test_lgb + opt_w[2]*test_cat + opt_w[3]*test_bridge, 1.0, 5.0)

# 7. Generate Submissions
sub_v11 = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_v11, 4)
})
sub_v11["_id"] = sub_v11["filename"].str.extract(r"(\d+)").astype(int)
sub_v11 = sub_v11.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)
sub_v11.to_csv("submission_v11_dual_transformer_titan.csv", index=False)

# Load existing benchmark champions (0.4846 and 0.5026)
v10_blend = pd.read_csv("submission_v10_master_blend.csv") # Leaderboard: 0.4846
v9_tri = pd.read_csv("submission_v9_tri_blend.csv")       # Leaderboard: 0.5026

# Key-aligned Titan Master Blend:
# 60% V11 Titan (MPNet-768 + MiniLM-384) + 30% V10 Master Blend (0.4846) + 10% V9 Tri (0.5026)
merged_all = sub_v11.merge(v10_blend, on="filename", suffixes=("_v11", "_v10")).merge(v9_tri, on="filename")
merged_all.rename(columns={"label": "label_v9"}, inplace=True)

merged_all["titan_blend"] = np.round(
    0.60 * merged_all["label_v11"] + 0.30 * merged_all["label_v10"] + 0.10 * merged_all["label_v9"], 4
)

sub_titan = merged_all[["filename", "titan_blend"]].rename(columns={"titan_blend": "label"})
sub_titan.to_csv("submission_v11_titan_blend.csv", index=False)
sub_titan.to_csv("submission.csv", index=False)

print("\n5/5 [SUCCESS] Generated and saved:")
print(f"1. submission_v11_dual_transformer_titan.csv -> Mean: {sub_v11['label'].mean():.4f}, Std: {sub_v11['label'].std():.4f}")
print(f"2. submission_v11_titan_blend.csv (60% V11 + 30% V10 [0.4846] + 10% V9 [0.5026]) -> Mean: {sub_titan['label'].mean():.4f}, Std: {sub_titan['label'].std():.4f}")
print("\nSample Titan Blend Predictions:\n", sub_titan.head(10))
print("\nSummary Statistics:\n", sub_titan["label"].describe())
