"""
Evaluation Metrics Module for SHL Grammar Scoring Engine.
Computes competition leaderboard metrics:
- Pearson Correlation Coefficient (r)
- Root Mean Squared Error (RMSE)
- Spearman Rank Correlation (rho)
- Mean Absolute Error (MAE)
"""

import numpy as np
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes all task-relevant evaluation metrics.
    Ensures safe handling of constant predictions or NaNs.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Clip predictions to competition valid range [0.0, 5.0]
    y_pred = np.clip(y_pred, 0.0, 5.0)

    # RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Pearson Correlation Coefficient
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    if std_true < 1e-7 or std_pred < 1e-7:
        pearson_r = 0.0
    else:
        cov = np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)))
        pearson_r = float(cov / (std_true * std_pred))

    # Spearman Rank Correlation
    rank_true = np.argsort(np.argsort(y_true))
    rank_pred = np.argsort(np.argsort(y_pred))
    std_rank_true = np.std(rank_true)
    std_rank_pred = np.std(rank_pred)
    if std_rank_true < 1e-7 or std_rank_pred < 1e-7:
        spearman_rho = 0.0
    else:
        cov_rank = np.mean(
            (rank_true - np.mean(rank_true)) * (rank_pred - np.mean(rank_pred))
        )
        spearman_rho = float(cov_rank / (std_rank_true * std_rank_pred))

    return {
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "mae": mae,
    }


def format_metrics_report(metrics: Dict[str, float], split_name: str = "Validation") -> str:
    """Returns a formatted tabular string of the metrics."""
    return (
        f"[{split_name} Evaluation]\n"
        f"  - Pearson Correlation (r) : {metrics['pearson_r']:.4f}\n"
        f"  - RMSE                     : {metrics['rmse']:.4f}\n"
        f"  - Spearman Rank (rho)     : {metrics['spearman_rho']:.4f}\n"
        f"  - MAE                      : {metrics['mae']:.4f}"
    )
