"""
Model Training, Cross-Validation, and Submission Generation Module.
Combines Acoustic, Fluency, and Linguistic/Grammar features.
Implements:
- 5-Fold Stratified Cross-Validation (binned targets)
- Ridge Regressor with standard scaling and regularized alpha search
- LightGBM Regressor optimized for small multimodal tabular features
- Weighted Ensemble blending
- Comprehensive evaluation: Pearson r, RMSE, Spearman rho, MAE
- Compulsory metric: Computes and logs Training Data RMSE
- Generates test set predictions bounded in [0.0, 5.0]
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from lightgbm import LGBMRegressor
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from metrics import compute_metrics, format_metrics_report
except ImportError:
    from src.metrics import compute_metrics, format_metrics_report


FEATURE_COLS = [
    # Acoustic Features
    "rms_energy",
    "max_amplitude",
    "mean_amplitude",
    "silence_ratio",
    "speech_ratio",
    "speech_duration_sec",
    "energy_std",
    "energy_skew",
    "zero_crossing_rate",
    "is_silent_or_corrupt",
    # Linguistic & Syntactic Features
    "num_words",
    "num_unique_words",
    "ttr",
    "guiraud_index",
    "mean_word_length",
    "polysyllable_ratio",
    "num_sentences",
    "mean_sentence_length",
    "std_sentence_length",
    "subordination_count",
    "subordination_ratio",
    "relative_pronoun_count",
    "complex_modal_count",
    "filler_count",
    "filler_ratio",
    "repetition_count",
    "repetition_ratio",
    "self_correction_count",
    "incomplete_sentence_ratio",
    "words_per_minute",
    "speech_rate",
    # J-Eval / G-Eval LLM Rubric Judge Features
    "judge_rubric_mos",
    "grammar_accuracy_score",
    "syntax_score",
    "sentence_completion_score",
    # Interaction Features
    "fluency_score",
    "complexity_score",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-specific interaction features combining acoustic & syntactic markers."""
    df = df.copy()

    # Ensure all feature columns exist and fill missing values safely
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    wpm = df["words_per_minute"].clip(lower=0.0)
    filler_ratio = df["filler_ratio"].clip(0.0, 1.0)
    incomplete_ratio = df["incomplete_sentence_ratio"].clip(0.0, 1.0)
    df["fluency_score"] = wpm * (1.0 - filler_ratio) * (1.0 - incomplete_ratio * 0.5)

    sub_ratio = df["subordination_ratio"].clip(lower=0.0)
    rel_pronouns = df["relative_pronoun_count"] / np.maximum(1.0, df["num_sentences"])
    df["syntax_score"] = sub_ratio + rel_pronouns

    poly_ratio = df["polysyllable_ratio"].clip(lower=0.0)
    guiraud = df["guiraud_index"].clip(lower=0.0)
    df["complexity_score"] = poly_ratio * guiraud

    return df


def train_evaluate_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Runs full 5-Fold Cross-Validation, calculates OOF and Train metrics, and predicts on test set.
    """
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    # Bin targets for stratification
    # Bins: [0, 1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0]
    bins = [-0.1, 1.5, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7, 5.1]
    target_bins = pd.cut(train_df["label"], bins=bins, labels=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X = train_df[FEATURE_COLS].values
    y = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values

    oof_preds_ridge = np.zeros(len(train_df))
    oof_preds_lgb = np.zeros(len(train_df))
    test_preds_ridge = np.zeros(len(test_df))
    test_preds_lgb = np.zeros(len(test_df))

    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"Beginning {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_bins)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]

        # 1. Ridge Regressor with standard scaler
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)

        ridge = RidgeCV(alphas=np.logspace(-2, 3, 50))
        ridge.fit(X_tr_sc, y_tr)
        pred_val_ridge = ridge.predict(X_va_sc)
        pred_test_ridge = ridge.predict(X_te_sc)

        oof_preds_ridge[val_idx] = pred_val_ridge
        test_preds_ridge += pred_test_ridge / n_splits

        # 2. LightGBM Regressor
        lgb = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=12,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state + fold,
            verbose=-1,
        )
        lgb.fit(X_tr, y_tr)
        pred_val_lgb = lgb.predict(X_va)
        pred_test_lgb = lgb.predict(X_test)

        oof_preds_lgb[val_idx] = pred_val_lgb
        test_preds_lgb += pred_test_lgb / n_splits

        # Fold blend (50/50)
        pred_val_blend = 0.5 * pred_val_ridge + 0.5 * pred_val_lgb
        f_metrics = compute_metrics(y_val, pred_val_blend)
        fold_metrics.append(f_metrics)
        print(f"Fold {fold + 1} -> Pearson r: {f_metrics['pearson_r']:.4f} | RMSE: {f_metrics['rmse']:.4f} | MAE: {f_metrics['mae']:.4f}")

    # Optimal ensemble weights via grid search on OOF
    best_weight = 0.5
    best_oof_rmse = float("inf")
    for w in np.linspace(0.0, 1.0, 101):
        blended = w * oof_preds_ridge + (1.0 - w) * oof_preds_lgb
        blended_clipped = np.clip(blended, 0.0, 5.0)
        rmse_val = np.sqrt(np.mean((y - blended_clipped) ** 2))
        if rmse_val < best_oof_rmse:
            best_oof_rmse = rmse_val
            best_weight = w

    oof_preds_final = np.clip(best_weight * oof_preds_ridge + (1.0 - best_weight) * oof_preds_lgb, 0.0, 5.0)
    test_preds_final = np.clip(best_weight * test_preds_ridge + (1.0 - best_weight) * test_preds_lgb, 0.0, 5.0)

    # 1. Out-of-Fold (OOF) Metrics
    oof_metrics = compute_metrics(y, oof_preds_final)

    # 2. COMPULSORY: Train RMSE (Fit full model or average fold predictions on train)
    # Fit full model on all 769 train samples to get exact Training Data RMSE
    scaler_full = StandardScaler()
    X_full_sc = scaler_full.fit_transform(X)
    ridge_full = Ridge(alpha=ridge.alpha_).fit(X_full_sc, y)
    lgb_full = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    ).fit(X, y)

    full_train_pred_ridge = ridge_full.predict(X_full_sc)
    full_train_pred_lgb = lgb_full.predict(X)
    full_train_pred = np.clip(best_weight * full_train_pred_ridge + (1.0 - best_weight) * full_train_pred_lgb, 0.0, 5.0)

    train_metrics = compute_metrics(y, full_train_pred)

    print(f"\n{'='*60}")
    print(f"COMPULSORY EVALUATION METRICS SUMMARY")
    print(f"{'='*60}")
    print(format_metrics_report(train_metrics, split_name="TRAINING DATA (Compulsory)"))
    print()
    print(format_metrics_report(oof_metrics, split_name="OUT-OF-FOLD (OOF) VALIDATION"))
    print(f"Optimal Ensemble Weight (Ridge vs LightGBM): {best_weight:.2f} / {1.0 - best_weight:.2f}")

    # Build submission dataframe (sorted naturally by audio index audio_0.wav to audio_215.wav)
    sub_df = pd.DataFrame({
        "filename": test_df["filename"],
        "label": np.round(test_preds_final, 4),
    })
    sub_df["_sort_id"] = sub_df["filename"].str.extract(r"(\d+)").astype(int)
    sub_df = sub_df.sort_values("_sort_id").drop(columns=["_sort_id"]).reset_index(drop=True)

    results_summary = {
        "train_metrics": train_metrics,
        "oof_metrics": oof_metrics,
        "fold_metrics": fold_metrics,
        "best_weight": float(best_weight),
        "oof_predictions": oof_preds_final.tolist(),
        "train_predictions": full_train_pred.tolist(),
        "y_true": y.tolist(),
        "feature_names": FEATURE_COLS,
        "ridge_coefs": ridge_full.coef_.tolist(),
        "lgb_importances": lgb_full.feature_importances_.tolist(),
    }

    return sub_df, results_summary
