"""
evaluate.py — Evaluate the trained win-probability logistic regression.

Loads the saved model and rolling predictions, then produces:

Console
    Log-loss, Brier score, accuracy.

Figures (saved to outputs/figures/)
    calibration_curve.png       Predicted probability vs actual win rate.
    feature_coefficients.png    Model coefficients sorted by magnitude.
    elo_time_series.png         Per-team Elo rating progression (small multiples).
    elo_all_teams.png           All 30 teams' Elo ratings overlaid on one chart.
    rolling_accuracy.png        Prediction accuracy throughout the season.
    roc_curve.png               ROC curve with AUC.
    confidence_histogram.png    Distribution of predicted probabilities.
    accuracy_by_confidence.png  Accuracy binned by model confidence level.
    team_accuracy.png           Per-team prediction accuracy ranked bar chart.

Run from the project root::

    python src/models/evaluate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

_HERE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_HERE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERE_PROJECT_ROOT))

from src.utils.io import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT, read_processed  # noqa: E402
from src.models.train import MODEL_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dark theme constants
# ---------------------------------------------------------------------------

BG    = "#131722"
PANEL = "#1e222d"
TEXT  = "#d1d4dc"
GRID  = "#2a2e39"
BLUE  = "#2962ff"
RED   = "#ff3c00"


def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=9, length=0)
    ax.grid(which="major", color=GRID, linewidth=0.6, linestyle="-", zorder=1)
    ax.set_axisbelow(True)


def _style_fig(fig) -> None:
    fig.patch.set_facecolor(BG)


def _legend(ax, **kwargs):
    ax.legend(
        frameon=True, facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=9,
        **kwargs,
    )


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
    bins = np.linspace(0, 1, n_bins + 1)
    indices = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)

    bucket_pred   = np.zeros(n_bins)
    bucket_actual = np.zeros(n_bins)
    bucket_count  = np.zeros(n_bins)

    for i in range(n_bins):
        mask = indices == i
        if mask.sum() > 0:
            bucket_pred[i]   = y_proba[mask].mean()
            bucket_actual[i] = y_true.values[mask].mean()
            bucket_count[i]  = mask.sum()

    non_empty = bucket_count > 0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    ax.plot([0, 1], [0, 1], color=TEXT, linewidth=1, linestyle="--", label="Perfect calibration")
    sc = ax.scatter(
        bucket_pred[non_empty],
        bucket_actual[non_empty],
        c=bucket_count[non_empty],
        s=70,
        zorder=3,
        cmap="Blues",
        edgecolors=BLUE,
        linewidths=0.8,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Games in bucket", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(colors=TEXT, length=0)
    cbar.ax.set_facecolor(PANEL)
    cbar.outline.set_edgecolor(GRID)

    ax.set_xlabel("Mean predicted probability", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Actual win rate", color=TEXT, fontsize=9, labelpad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _legend(ax, loc="upper left")
    fig.suptitle("Calibration curve — logistic regression", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Calibration curve → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Feature coefficients
# ---------------------------------------------------------------------------

def plot_feature_coefficients(
    model,
    out_path: Path = FIGURES_DIR / "feature_coefficients.png",
) -> None:
    clf   = model.named_steps["clf"]
    coefs = clf.coef_[0]

    order           = np.argsort(np.abs(coefs))[::-1]
    sorted_features = [MODEL_FEATURES[i] for i in order]
    sorted_coefs    = coefs[order]

    colors = [RED if c < 0 else BLUE for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(MODEL_FEATURES))), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    ax.barh(sorted_features[::-1], sorted_coefs[::-1], color=colors[::-1], zorder=2)
    ax.axvline(0, color=TEXT, linewidth=0.8)
    ax.set_xlabel("Coefficient value", color=TEXT, fontsize=9, labelpad=8)
    fig.suptitle("Logistic regression feature coefficients", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Feature coefficients → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Elo time series — small multiples
# ---------------------------------------------------------------------------

def plot_elo_time_series(
    elo_ratings: pd.DataFrame,
    team_features: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "elo_time_series.png",
) -> None:
    abbrev_map = (
        team_features[["team_id", "team_abbreviation"]]
        .drop_duplicates()
        .set_index("team_id")["team_abbreviation"]
    )
    elo = elo_ratings.copy()
    elo["team_abbreviation"] = elo["team_id"].map(abbrev_map)
    elo["game_date"] = pd.to_datetime(elo["game_date"])

    teams = sorted(elo["team_abbreviation"].dropna().unique())
    ncols = 6
    nrows = (len(teams) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3), sharey=False, dpi=150)
    _style_fig(fig)
    axes = axes.flatten()

    for i, team in enumerate(teams):
        ax   = axes[i]
        data = elo[elo["team_abbreviation"] == team].sort_values("game_date")
        _style_ax(ax)
        ax.plot(data["game_date"], data["elo_pre"], linewidth=1.4, color=BLUE, zorder=2)
        ax.axhline(1500, color=GRID, linewidth=0.9, linestyle="--")
        ax.set_title(team, fontsize=9, fontweight="bold", color=TEXT)
        ax.tick_params(axis="x", rotation=45, labelsize=6, colors=TEXT)
        ax.tick_params(axis="y", labelsize=7, colors=TEXT)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    for j in range(len(teams), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Elo rating progression — 2024-25 NBA regular season",
        color=TEXT, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Elo time series → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Elo all teams — single overlaid chart
# ---------------------------------------------------------------------------

def plot_elo_all_teams(
    elo_ratings: pd.DataFrame,
    team_features: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "elo_all_teams.png",
) -> None:
    """All 30 teams' Elo ratings on a single axes, colour-coded by team."""
    abbrev_map = (
        team_features[["team_id", "team_abbreviation"]]
        .drop_duplicates()
        .set_index("team_id")["team_abbreviation"]
    )
    elo = elo_ratings.copy()
    elo["team_abbreviation"] = elo["team_id"].map(abbrev_map)
    elo["game_date"] = pd.to_datetime(elo["game_date"])

    teams = sorted(elo["team_abbreviation"].dropna().unique())
    n = len(teams)

    # Build a 30-colour palette from tab20 + tab20b
    tab20  = plt.cm.tab20.colors   # type: ignore[attr-defined]
    tab20b = plt.cm.tab20b.colors  # type: ignore[attr-defined]
    palette = list(tab20) + list(tab20b)
    team_colors = {team: palette[i % len(palette)] for i, team in enumerate(teams)}

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    for team in teams:
        data = elo[elo["team_abbreviation"] == team].sort_values("game_date")
        ax.plot(
            data["game_date"], data["elo_pre"],
            linewidth=1.0, alpha=0.85,
            color=team_colors[team],
            label=team,
            zorder=2,
        )

    ax.axhline(1500, color=TEXT, linewidth=0.7, linestyle="--", alpha=0.4, zorder=1)

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.label.set_color(TEXT)

    ax.set_xlabel("Date", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Elo rating", color=TEXT, fontsize=9, labelpad=8)

    # Legend outside the axes in 3 columns to fit 30 teams
    leg = ax.legend(
        frameon=True, facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=7,
        ncol=3, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
    )

    fig.suptitle(
        "Elo ratings — all teams — 2024-25 NBA regular season",
        color=TEXT, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Elo all teams → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Rolling accuracy
# ---------------------------------------------------------------------------

def plot_rolling_accuracy(
    rolling_preds: pd.DataFrame,
    window: int = 15,
    out_path: Path = FIGURES_DIR / "rolling_accuracy.png",
) -> None:
    df = rolling_preds.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    df["correct"] = (df["predicted_label"] == df["home_win"]).astype(int)

    df["week"] = df["game_date"].dt.to_period("W").dt.start_time
    weekly = df.groupby("week")["correct"].agg(["mean", "count"]).reset_index()

    df["rolling_acc"] = df["correct"].rolling(window, min_periods=max(1, window // 2)).mean()
    season_mean = df["correct"].mean()

    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    ax.bar(
        weekly["week"], weekly["mean"],
        width=5, alpha=0.25, color=BLUE, label="Weekly accuracy", zorder=2,
    )
    ax.plot(
        df["game_date"], df["rolling_acc"],
        color=BLUE, linewidth=1.8, label=f"{window}-game rolling accuracy", zorder=3,
    )
    ax.axhline(
        season_mean, color=RED, linewidth=1.2, linestyle="--",
        label=f"Season mean  {season_mean:.3f}", zorder=3,
    )

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Date", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Accuracy", color=TEXT, fontsize=9, labelpad=8)
    _legend(ax, loc="lower right")
    fig.suptitle("Prediction accuracy throughout the season", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Rolling accuracy → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    out_path: Path = FIGURES_DIR / "roc_curve.png",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    ax.plot(fpr, tpr, color=BLUE, linewidth=1.8, label=f"AUC = {auc:.4f}", zorder=3)
    ax.plot([0, 1], [0, 1], color=TEXT, linewidth=1, linestyle="--", label="Random classifier", zorder=2)
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE, zorder=1)

    ax.set_xlabel("False positive rate", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("True positive rate", color=TEXT, fontsize=9, labelpad=8)
    _legend(ax, loc="lower right")
    fig.suptitle("ROC curve — logistic regression", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("ROC curve → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Confidence histogram
# ---------------------------------------------------------------------------

def plot_confidence_histogram(
    y_true: pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 25,
    out_path: Path = FIGURES_DIR / "confidence_histogram.png",
) -> None:
    y_true_arr = np.asarray(y_true)
    wins   = y_proba[y_true_arr == 1]
    losses = y_proba[y_true_arr == 0]
    bins   = np.linspace(0, 1, n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    ax.hist(wins,   bins=bins, alpha=0.55, color=BLUE, label="Home win",  zorder=2)
    ax.hist(losses, bins=bins, alpha=0.55, color=RED,  label="Home loss", zorder=2)
    ax.axvline(0.5, color=TEXT, linewidth=0.9, linestyle="--", zorder=3)

    ax.set_xlabel("Predicted home-win probability", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Games", color=TEXT, fontsize=9, labelpad=8)
    _legend(ax)
    fig.suptitle(
        "Distribution of predicted probabilities by actual outcome",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Confidence histogram → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Accuracy by confidence band
# ---------------------------------------------------------------------------

def plot_accuracy_by_confidence(
    y_true: pd.Series,
    y_proba: np.ndarray,
    out_path: Path = FIGURES_DIR / "accuracy_by_confidence.png",
) -> None:
    confidence = np.maximum(y_proba, 1 - y_proba)
    correct    = (np.round(y_proba) == np.asarray(y_true)).astype(int)

    edges  = np.arange(0.50, 1.01, 0.05)
    labels = [f"{e:.2f}–{e+0.05:.2f}" for e in edges[:-1]]
    indices = np.clip(np.digitize(confidence, edges) - 1, 0, len(edges) - 2)

    accs, counts = [], []
    for i in range(len(edges) - 1):
        mask = indices == i
        accs.append(correct[mask].mean() if mask.sum() > 0 else np.nan)
        counts.append(mask.sum())

    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=150)
    _style_fig(fig)
    _style_ax(ax1)

    ax2 = ax1.twinx()
    ax2.set_facecolor(PANEL)
    ax2.tick_params(colors=TEXT, labelsize=9, length=0)

    x = np.arange(len(labels))
    ax1.bar(x, counts, color=BLUE, alpha=0.2, label="Games", zorder=2)
    ax2.plot(x, accs, color=BLUE, marker="o", linewidth=1.8, markersize=5, label="Accuracy", zorder=3)
    ax2.axhline(correct.mean(), color=RED, linewidth=1, linestyle="--",
                label=f"Overall mean  {correct.mean():.3f}", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Confidence band  max(p, 1−p)", color=TEXT, fontsize=9, labelpad=8)
    ax1.set_ylabel("Games", color=TEXT, fontsize=9, labelpad=8)
    ax2.set_ylabel("Accuracy", color=TEXT, fontsize=9, labelpad=8)
    ax2.set_ylim(0.4, 1.0)

    # Combine legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=True, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, fontsize=9, loc="upper left")

    fig.suptitle("Accuracy by model confidence band", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Accuracy by confidence → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Team-level accuracy
# ---------------------------------------------------------------------------

def plot_team_accuracy(
    rolling_preds: pd.DataFrame,
    team_features: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "team_accuracy.png",
) -> None:
    abbrev_map = (
        team_features[["team_id", "team_abbreviation"]]
        .drop_duplicates()
        .set_index("team_id")["team_abbreviation"]
    )

    preds = rolling_preds.copy()
    preds["correct"] = (preds["predicted_label"] == preds["home_win"]).astype(int)

    home = preds[["home_team_id", "correct"]].rename(columns={"home_team_id": "team_id"})
    away = preds[["away_team_id", "correct"]].rename(columns={"away_team_id": "team_id"})
    combined = pd.concat([home, away], ignore_index=True)

    team_acc = (
        combined.groupby("team_id")["correct"]
        .agg(accuracy="mean", games="count")
        .reset_index()
    )
    team_acc["team"] = team_acc["team_id"].map(abbrev_map)
    team_acc = team_acc.sort_values("accuracy", ascending=False).reset_index(drop=True)

    overall = preds["correct"].mean()
    colors  = [BLUE if a >= overall else RED for a in team_acc["accuracy"]]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    _style_fig(fig)
    _style_ax(ax)

    ax.bar(team_acc["team"], team_acc["accuracy"], color=colors, zorder=2)
    ax.axhline(overall, color=TEXT, linewidth=1, linestyle="--",
               label=f"Overall mean  {overall:.3f}", zorder=3)

    ax.set_xlabel("Team", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Accuracy", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    _legend(ax)
    fig.suptitle(
        "Per-team prediction accuracy (home + away games combined)",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Team accuracy → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    model_path = MODELS_DIR / "win_probability_logreg.joblib"
    pred_path  = MODELS_DIR / "rolling_predictions.parquet"

    log.info("Loading model from %s ...", model_path.relative_to(PROJECT_ROOT))
    model = joblib.load(model_path)

    log.info("Loading rolling predictions from %s ...", pred_path.relative_to(PROJECT_ROOT))
    preds = pd.read_parquet(pred_path)

    log.info("Loading processed features ...")
    team_features = read_processed("team_features.parquet")
    elo_ratings   = read_processed("elo_ratings.parquet")

    y_true = preds["home_win"]
    y_proba = preds["predicted_proba"].values
    y_pred  = preds["predicted_label"].values

    print_metrics(y_true, y_proba, y_pred)
    plot_calibration_curve(y_true, y_proba)
    plot_feature_coefficients(model)
    plot_elo_time_series(elo_ratings, team_features)
    plot_elo_all_teams(elo_ratings, team_features)
    plot_rolling_accuracy(preds)
    plot_roc_curve(y_true, y_proba)
    plot_confidence_histogram(y_true, y_proba)
    plot_accuracy_by_confidence(y_true, y_proba)
    plot_team_accuracy(preds, team_features)

    print("\nDone.")


if __name__ == "__main__":
    main()
