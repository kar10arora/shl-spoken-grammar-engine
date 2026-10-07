import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr

# 1. Load data
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

# 2. Build Text N-Gram Representation (Word 1-2 + Char 3-5)
print("Fitting TF-IDF Word & Char N-Grams...")
tfidf_word = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, min_df=2)
tfidf_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=1500, min_df=3)

all_train_text = train_trans["transcript"].fillna("").values
all_test_text = test_trans["transcript"].fillna("").values

X_w_tr = tfidf_word.fit_transform(all_train_text).toarray()
X_w_te = tfidf_word.transform(all_test_text).toarray()

X_c_tr = tfidf_char.fit_transform(all_train_text).toarray()
X_c_te = tfidf_char.transform(all_test_text).toarray()

X_text_tr = np.hstack([X_w_tr, X_c_tr])
X_text_te = np.hstack([X_w_te, X_c_te])

y = train_df["label"].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

oof_text_ridge = np.zeros(len(y))
test_text_ridge = np.zeros(len(test_trans))

for tr, val in skf.split(X_text_tr, target_bins):
    m = RidgeCV(alphas=np.logspace(-2, 3, 50))
    m.fit(X_text_tr[tr], y[tr])
    oof_text_ridge[val] = m.predict(X_text_tr[val])
    test_text_ridge += m.predict(X_text_te) / 5.0

print(f"Text Ridge alone -> Pearson r: {pearsonr(y, oof_text_ridge)[0]:.4f} | RMSE: {np.sqrt(np.mean((y - oof_text_ridge)**2)):.4f}")

train_trans["text_ngram_score"] = oof_text_ridge
test_trans["text_ngram_score"] = test_text_ridge

# 3. Merge all modalities
ling_cols = [c for c in train_ling.columns if c not in ["label", "duration_sec", "speech_duration_sec"]]
judge_cols = ["filename", "judge_rubric_mos", "grammar_accuracy_score", "syntax_score", "sentence_completion_score"]

train_full = train_audio.merge(train_ling[ling_cols], on="filename").merge(train_judge[judge_cols], on="filename").merge(train_trans[["filename", "text_ngram_score"]], on="filename")
test_full = test_audio.merge(test_ling[ling_cols], on="filename", how="left").merge(test_judge[judge_cols], on="filename", how="left").merge(test_trans[["filename", "text_ngram_score"]], on="filename", how="left")

for col in judge_cols[1:]:
    test_full[col] = test_full[col].fillna(3.0)

# Add interactions
def add_features(df):
    df = df.copy()
    df["judge_x_speech_ratio"] = df["judge_rubric_mos"] * df["speech_ratio"]
    df["judge_x_ttr"] = df["judge_rubric_mos"] * df["ttr"]
    df["judge_x_text_ngram"] = df["judge_rubric_mos"] * df["text_ngram_score"]
    df["fluency_index"] = df["words_per_minute"] / (1.0 + df["filler_count"] + df["repetition_count"])
    return df

train_full = add_features(train_full)
test_full = add_features(test_full)

FEATURE_COLS = [c for c in train_full.columns if c not in ["filename", "label"]]
print(f"Total Unified Multimodal Features: {len(FEATURE_COLS)}")

X = train_full[FEATURE_COLS].values
X_test = test_full[FEATURE_COLS].values

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(test_full))
oof_ridge = np.zeros(len(y))
test_ridge = np.zeros(len(test_full))

for tr, val in skf.split(X, target_bins):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X[tr])
    X_va_sc = scaler.transform(X[val])
    X_te_sc = scaler.transform(X_test)
    
    # Ridge
    r_model = RidgeCV(alphas=np.logspace(-2, 3, 50))
    r_model.fit(X_tr_sc, y[tr])
    oof_ridge[val] = r_model.predict(X_va_sc)
    test_ridge += r_model.predict(X_te_sc) / 5.0
    
    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=160,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        min_child_samples=16,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1
    )
    lgb.fit(X[tr], y[tr])
    oof_lgb[val] = lgb.predict(X[val])
    test_lgb += lgb.predict(X_test) / 5.0

print(f"LightGBM -> Pearson r: {pearsonr(y, oof_lgb)[0]:.4f} | RMSE: {np.sqrt(np.mean((y - oof_lgb)**2)):.4f}")
print(f"Ridge    -> Pearson r: {pearsonr(y, oof_ridge)[0]:.4f} | RMSE: {np.sqrt(np.mean((y - oof_ridge)**2)):.4f}")

# Optimize blend weight
best_w = 0.5
best_rmse = float("inf")
for w in np.linspace(0.0, 1.0, 101):
    blend = np.clip(w * oof_ridge + (1 - w) * oof_lgb, 0.0, 5.0)
    rmse = np.sqrt(np.mean((y - blend)**2))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

print(f"Optimal Ridge Weight: {best_w:.2f} | Blended RMSE: {best_rmse:.4f}")
oof_final = np.clip(best_w * oof_ridge + (1 - best_w) * oof_lgb, 0.0, 5.0)
r_final, _ = pearsonr(y, oof_final)
print(f"FINAL BLEND -> Pearson r: {r_final:.4f} | RMSE: {best_rmse:.4f}")

test_final = np.clip(best_w * test_ridge + (1 - best_w) * test_lgb, 0.0, 5.0)

sub = pd.DataFrame({
    "filename": test_full["filename"],
    "label": np.round(test_final, 4)
})
sub["_id"] = sub["filename"].str.extract(r"(\d+)").astype(int)
sub = sub.sort_values("_id").drop(columns=["_id"]).reset_index(drop=True)

sub.to_csv("submission_ngram_multimodal.csv", index=False)
print("Saved submission_ngram_multimodal.csv!")
print(sub.head(10))
