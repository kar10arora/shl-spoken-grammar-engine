import os
import sys
import wave
import numpy as np
import pandas as pd
import nltk
import textstat
from scipy import signal
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV, Ridge, BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

print("="*60)
print("Building V4 Hyper Engine: Spectral Acoustics + POS Syntax + Multi-Ngrams + Meta-Ensemble")
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

# 2. Extract Spectral Audio Features (Spectral Centroid & Rolloff)
def extract_spectral(audio_dir, file_list):
    records = []
    for fname in file_list:
        fpath = os.path.join(audio_dir, fname)
        if not os.path.exists(fpath):
            records.append({"filename": fname, "spec_centroid_mean": 1000.0, "spec_centroid_std": 400.0, "spec_rolloff_mean": 1800.0, "spec_rolloff_std": 600.0})
            continue
        try:
            with wave.open(fpath, "rb") as wf:
                sr = wf.getframerate()
                nframes = wf.getnframes()
                audio = np.frombuffer(wf.readframes(nframes), dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 1000 or np.max(np.abs(audio)) < 1e-4:
                records.append({"filename": fname, "spec_centroid_mean": 0.0, "spec_centroid_std": 0.0, "spec_rolloff_mean": 0.0, "spec_rolloff_std": 0.0})
                continue
            # Downsample if needed or take STFT with 512 window
            f, t, Zxx = signal.stft(audio, fs=sr, nperseg=512, noverlap=256)
            mag = np.abs(Zxx) + 1e-10
            centroid = np.sum(f[:, None] * mag, axis=0) / np.sum(mag, axis=0)
            cumsum_mag = np.cumsum(mag, axis=0)
            cutoff = 0.85 * cumsum_mag[-1, :]
            rolloff = f[np.argmax(cumsum_mag >= cutoff, axis=0)]
            records.append({
                "filename": fname,
                "spec_centroid_mean": float(np.mean(centroid)),
                "spec_centroid_std": float(np.std(centroid)),
                "spec_rolloff_mean": float(np.mean(rolloff)),
                "spec_rolloff_std": float(np.std(rolloff))
            })
        except Exception:
            records.append({"filename": fname, "spec_centroid_mean": 1000.0, "spec_centroid_std": 400.0, "spec_rolloff_mean": 1800.0, "spec_rolloff_std": 600.0})
    return pd.DataFrame(records)

print("[1/5] Extracting Spectral Audio Features (Centroid & Rolloff)...")
spec_tr = extract_spectral("shl-hiring-assessment-2026/Dataset_Final/train", train_audio["filename"].tolist())
spec_te = extract_spectral("shl-hiring-assessment-2026/Dataset_Final/test", test_audio["filename"].tolist())

# 3. Extract Readability Indices
print("[2/5] Extracting Readability & Syntactic Maturity Metrics...")
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
    res = pd.DataFrame(records)
    res["filename"] = df["filename"].values
    return res

read_tr = extract_readability(train_trans)
read_te = extract_readability(test_trans)

# 4. Extract POS Tag Sequences & POS N-Grams
print("[3/5] Extracting Part-of-Speech Sequences & POS N-Grams...")
def get_pos_seqs(df):
    seqs = []
    for t in df["transcript"].fillna(""):
        if not t.strip():
            seqs.append("")
            continue
        words = nltk.word_tokenize(t)
        tags = nltk.pos_tag(words)
        seqs.append(" ".join([pos for w, pos in tags]))
    return seqs

pos_tr = get_pos_seqs(train_trans)
pos_te = get_pos_seqs(test_trans)

tfidf_pos = TfidfVectorizer(ngram_range=(1, 3), max_features=1200, min_df=3, sublinear_tf=True)
X_pos_tr = tfidf_pos.fit_transform(pos_tr).toarray()
X_pos_te = tfidf_pos.transform(pos_te).toarray()

oof_pos_ridge = np.zeros(len(y))
test_pos_ridge = np.zeros(len(test_trans))
for tr, val in skf.split(X_pos_tr, target_bins):
    m = RidgeCV(alphas=np.logspace(-2, 3, 50))
    m.fit(X_pos_tr[tr], y[tr])
    oof_pos_ridge[val] = m.predict(X_pos_tr[val])
    test_pos_ridge += m.predict(X_pos_te) / 5.0

print(f"  * POS N-Gram Ridge alone -> Pearson r: {pearsonr(y, oof_pos_ridge)[0]:.4f} | RMSE: {np.sqrt(np.mean((y - oof_pos_ridge)**2)):.4f}")

# 5. Extract Multi-Gram Word & Character Features + SVD
print("[4/5] Extracting Word (1-2) + Character (3-5) N-Grams & 20 SVD Manifolds...")
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

print(f"  * Text N-Gram Ridge alone -> Pearson r: {pearsonr(y, oof_text_ridge)[0]:.4f} | RMSE: {np.sqrt(np.mean((y - oof_text_ridge)**2)):.4f}")

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

train_trans["pos_ngram_score"] = oof_pos_ridge
test_trans["pos_ngram_score"] = test_pos_ridge

# 6. Unify All Modalities
ling_cols = [c for c in train_ling.columns if c not in ["label", "duration_sec", "speech_duration_sec"]]
judge_cols = ["filename", "judge_rubric_mos", "grammar_accuracy_score", "syntax_score", "sentence_completion_score"]

train_full = train_audio.merge(train_ling[ling_cols], on="filename").merge(train_judge[judge_cols], on="filename").merge(train_trans[["filename", "text_ngram_score", "pos_ngram_score"]], on="filename").merge(read_tr, on="filename").merge(spec_tr, on="filename").merge(svd_tr_df, on="filename")

test_full = test_audio.merge(test_ling[ling_cols], on="filename", how="left").merge(test_judge[judge_cols], on="filename", how="left").merge(test_trans[["filename", "text_ngram_score", "pos_ngram_score"]], on="filename", how="left").merge(read_te, on="filename", how="left").merge(spec_te, on="filename", how="left").merge(svd_te_df, on="filename", how="left")

for col in judge_cols[1:]:
    test_full[col] = test_full[col].fillna(3.0)

# Domain Interactions
def add_features(df):
    df = df.copy()
    df["judge_x_speech_ratio"] = df["judge_rubric_mos"] * df["speech_ratio"]
    df["judge_x_ttr"] = df["judge_rubric_mos"] * df["ttr"]
    df["judge_x_text_ngram"] = df["judge_rubric_mos"] * df["text_ngram_score"]
    df["judge_x_pos_ngram"] = df["judge_rubric_mos"] * df["pos_ngram_score"]
    df["text_x_pos_ngram"] = df["text_ngram_score"] * df["pos_ngram_score"]
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

# 7. Multi-Model 5-Fold Stratified Training
print("\n[5/5] Training Diverse Models & Optimizing Stacking Meta-Weights...")
oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(test_full))

oof_cat = np.zeros(len(y))
test_cat = np.zeros(len(test_full))

oof_xgb = np.zeros(len(y))
test_xgb = np.zeros(len(test_full))

oof_bridge = np.zeros(len(y))
test_bridge = np.zeros(len(test_full))

for tr, val in skf.split(X, target_bins):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X[tr])
    X_va_sc = scaler.transform(X[val])
    X_te_sc = scaler.transform(X_test)

    # 1. XGBoost
    xgb_m = XGBRegressor(
        n_estimators=170,
        learning_rate=0.028,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42
    )
    xgb_m.fit(X[tr], y[tr])
    oof_xgb[val] = xgb_m.predict(X[val])
    test_xgb += xgb_m.predict(X_test) / 5.0

    # 2. LightGBM
    lgb_m = LGBMRegressor(
        n_estimators=190,
        learning_rate=0.025,
        max_depth=4,
        num_leaves=13,
        min_child_samples=18,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1
    )
    lgb_m.fit(X[tr], y[tr])
    oof_lgb[val] = lgb_m.predict(X[val])
    test_lgb += lgb_m.predict(X_test) / 5.0

    # 3. CatBoost
    cat_m = CatBoostRegressor(
        iterations=220,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0
    )
    cat_m.fit(X[tr], y[tr])
    oof_cat[val] = cat_m.predict(X[val])
    test_cat += cat_m.predict(X_test) / 5.0

    # 4. Bayesian Ridge
    br_m = BayesianRidge()
    br_m.fit(X_tr_sc, y[tr])
    oof_bridge[val] = br_m.predict(X_va_sc)
    test_bridge += br_m.predict(X_te_sc) / 5.0

def eval_m(preds, name):
    r, _ = pearsonr(y, preds)
    rmse = np.sqrt(np.mean((y - preds)**2))
    print(f"  * [{name:16s}] -> Pearson r: {r:.4f} | RMSE: {rmse:.4f}")
    return rmse

print("\n--- Individual Model OOF Scores ---")
eval_m(oof_xgb, "XGBoost")
eval_m(oof_lgb, "LightGBM")
eval_m(oof_cat, "CatBoost")
eval_m(oof_bridge, "Bayesian Ridge")

# Optimize 4-way blend weights directly for RMSE
def loss(w):
    w1, w2, w3, w4 = w
    pred = np.clip(w1*oof_xgb + w2*oof_lgb + w3*oof_cat + w4*oof_bridge, 0.0, 5.0)
    return np.sqrt(np.mean((y - pred)**2))

res = minimize(loss, [0.5, 0.2, 0.1, 0.2], bounds=[(0, 1)]*4, constraints={"type":"eq", "fun": lambda w: sum(w)-1.0})
opt_w = res.x
print(f"\nOptimal Ensemble Weights: XGBoost={opt_w[0]:.3f}, LGBM={opt_w[1]:.3f}, CatBoost={opt_w[2]:.3f}, BayesRidge={opt_w[3]:.3f}")

oof_final = np.clip(opt_w[0]*oof_xgb + opt_w[1]*oof_lgb + opt_w[2]*oof_cat + opt_w[3]*oof_bridge, 0.0, 5.0)
final_rmse = eval_m(oof_final, "V4 HYPER ENSEMBLE")

test_final = np.clip(opt_w[0]*test_xgb + opt_w[1]*test_lgb + opt_w[2]*test_cat + opt_w[3]*test_bridge, 0.0, 5.0)

sub = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_final, 4)
})
sub["_id"] = sub["filename"].str.extract(r"(\d+)").astype(int)
sub = sub.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)

sub.to_csv("submission_v4_hyper.csv", index=False)
sub.to_csv("submission.csv", index=False)
sub.to_csv("data/submission.csv", index=False)

print(f"\n[OK] Saved submission_v4_hyper.csv & submission.csv with {len(sub)} rows!")
print("Sample predictions:\n", sub.head())
print("Summary stats:\n", sub["label"].describe())
