"""
evaluate.py — Evaluate the trained win-probability logistic regression.

Loads the saved model and rolling predictions, then produces:
  * Console report: log-loss, Brier score, accuracy
  * outputs/figures/calibration_curve.png  — predicted probability vs actual
    win rate in 10 equal-width buckets
  * outputs/figures/feature_coefficients.png — model coefficients sorted by
    magnitude

Run from the project root::

    python src/models/evaluate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import FIGURES_DIR, MODELS_DIR  # noqa: E402
from src.models.train import MODEL_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(y_true: pd.Series, y_proba: np.ndarray, y_pred: np.ndarray) -> None:
    ll = log_loss(y_true, y_proba)
    bs = brier_score_loss(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    print("\n--- Evaluation metrics ---")
    print(f"  Log-loss    : {ll:.4f}")
    print(f"  Brier score : {bs:.4f}")
    print(f"  Accuracy    : {acc:.4f}")


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
    out_path: Path = FIGURES_DIR / "calibration_curve.png",
) -> None:
    """Bin predictions into equal-width buckets and plot mean predicted
    probability vs actual win rate.  A perfectly calibrated model lies on
    the diagonal.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    indices = np.digitize(y_proba, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    bucket_pred = np.zeros(n_bins)
    bucket_actual = np.zeros(n_bins)
    bucket_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = indices == i
        if mask.sum() > 0:
            bucket_pred[i] = y_proba[mask].mean()
            bucket_actual[i] = y_true.values[mask].mean()
            bucket_count[i] = mask.sum()

    non_empty = bucket_count > 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    sc = ax.scatter(
        bucket_pred[non_empty],
        bucket_actual[non_empty],
        c=bucket_count[non_empty],
        s=60,
        zorder=3,
        cmap="Blues",
        edgecolors="steelblue",
    )
    plt.colorbar(sc, ax=ax, label="Games in bucket")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration curve — logistic regression")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Calibration curve → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Feature coefficients
# ---------------------------------------------------------------------------

def plot_feature_coefficients(
    model,
    out_path: Path = FIGURES_DIR / "feature_coefficients.png",
) -> None:
    """Bar chart of logistic regression coefficients sorted by magnitude."""
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    order = np.argsort(np.abs(coefs))[::-1]
    sorted_features = [MODEL_FEATURES[i] for i in order]
    sorted_coefs = coefs[order]

    colors = ["#e05c5c" if c < 0 else "#5c9ee0" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(MODEL_FEATURES))))
    bars = ax.barh(sorted_features[::-1], sorted_coefs[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Logistic regression feature coefficients")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Feature coefficients → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    model_path = MODELS_DIR / "win_probability_logreg.joblib"
    pred_path = MODELS_DIR / "rolling_predictions.parquet"

    log.info("Loading model from %s ...", model_path.relative_to(PROJECT_ROOT))
    model = joblib.load(model_path)

    log.info("Loading test predictions from %s ...", pred_path.relative_to(PROJECT_ROOT))
    preds = pd.read_parquet(pred_path)

    y_true = preds["home_win"]
    y_proba = preds["predicted_proba"].values
    y_pred = preds["predicted_label"].values

    print_metrics(y_true, y_proba, y_pred)
    plot_calibration_curve(y_true, y_proba)
    plot_feature_coefficients(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
