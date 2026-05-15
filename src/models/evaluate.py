"""
evaluate.py — Evaluate a trained win-probability model.

Loads the saved model and rolling predictions, then produces:

Console
    Log-loss, Brier score, accuracy.

Figures
    outputs/figures/logreg/   (or xgboost/) — model-specific plots:
        calibration_curve.png       Predicted probability vs actual win rate.
        feature_coefficients.png    Logistic regression coefficients (logreg only).
        feature_importance.png      XGBoost feature importances (xgboost only).
        rolling_accuracy.png        Prediction accuracy throughout the season.
        roc_curve.png               ROC curve with AUC.
        confidence_histogram.png    Distribution of predicted probabilities.
        accuracy_by_confidence.png  Accuracy binned by model confidence level.
        team_accuracy.png           Per-team prediction accuracy ranked bar chart.
    outputs/figures/          — model-agnostic plots (shared):
        elo_time_series.png         Per-team Elo rating progression (small multiples).
        elo_all_teams.png           All 30 teams' Elo ratings overlaid on one chart.

Run from the project root::

    python src/models/evaluate.py                  # evaluate logreg
    python src/models/evaluate.py --model xgboost  # evaluate xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
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

from src.utils.io import FIGURES_DIR, MODELS_DIR, PROJECT_ROOT, SEASON, configure_logging, read_parquet, read_processed, season_api  # noqa: E402
from src.utils.style import PANEL, TEXT, GRID, BLUE, RED, NBA_TEAM_COLORS, style_ax, legend, styled_subplots, save_fig  # noqa: E402
from src.models.train import MODEL_FEATURES, XGBOOST_MODEL_FEATURES  # noqa: E402

FIGURES_LOGREG_DIR  = FIGURES_DIR / "logreg"
FIGURES_XGBOOST_DIR = FIGURES_DIR / "xgboost"

log = configure_logging("evaluate")


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

    fig, ax = styled_subplots((6, 6))

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
    legend(ax, loc="upper left")
    fig.suptitle("Calibration curve", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    save_fig(fig, out_path)
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

    fig, ax = styled_subplots((7, max(3, 0.5 * len(MODEL_FEATURES))))

    ax.barh(sorted_features[::-1], sorted_coefs[::-1], color=colors[::-1], zorder=2)
    ax.axvline(0, color=TEXT, linewidth=0.8)
    ax.set_xlabel("Coefficient value", color=TEXT, fontsize=9, labelpad=8)
    fig.suptitle("Logistic regression feature coefficients", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    save_fig(fig, out_path)
    log.info("Feature coefficients → %s", out_path.relative_to(PROJECT_ROOT))


def plot_feature_importance(
    model,
    out_path: Path = FIGURES_XGBOOST_DIR / "feature_importance.png",
) -> None:
    importances = model.feature_importances_

    order           = np.argsort(importances)[::-1]
    sorted_features = [XGBOOST_MODEL_FEATURES[i] for i in order]
    sorted_imps     = importances[order]

    fig, ax = styled_subplots((7, max(5, 0.35 * len(XGBOOST_MODEL_FEATURES))))

    ax.barh(sorted_features[::-1], sorted_imps[::-1], color=BLUE, zorder=2)
    ax.set_xlabel("Feature importance (gain)", color=TEXT, fontsize=9, labelpad=8)
    fig.suptitle("XGBoost feature importance", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    save_fig(fig, out_path)
    log.info("Feature importance → %s", out_path.relative_to(PROJECT_ROOT))


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

    fig, axes = styled_subplots((18, nrows * 3), nrows=nrows, ncols=ncols, sharey=False)
    axes = axes.flatten()

    for i, team in enumerate(teams):
        ax   = axes[i]
        data = elo[elo["team_abbreviation"] == team].sort_values("game_date")
        style_ax(ax)
        ax.plot(data["game_date"], data["elo_pre"], linewidth=1.4, color=BLUE, zorder=2)
        ax.axhline(1500, color=GRID, linewidth=0.9, linestyle="--")
        ax.set_title(team, fontsize=9, fontweight="bold", color=TEXT)
        ax.tick_params(axis="x", rotation=45, labelsize=6, colors=TEXT)
        ax.tick_params(axis="y", labelsize=7, colors=TEXT)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    for j in range(len(teams), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Elo rating progression — {season_api(SEASON)} NBA regular season",
        color=TEXT, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout()

    save_fig(fig, out_path)
    log.info("Elo time series → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Elo all teams — single overlaid chart
# ---------------------------------------------------------------------------

def plot_elo_all_teams(
    elo_ratings: pd.DataFrame,
    team_features: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "elo_all_teams.png",
    top_n: int = 10,
) -> None:
    """Top-N teams by final Elo rating on a single axes, coloured by franchise."""
    abbrev_map = (
        team_features[["team_id", "team_abbreviation"]]
        .drop_duplicates()
        .set_index("team_id")["team_abbreviation"]
    )
    elo = elo_ratings.copy()
    elo["team_abbreviation"] = elo["team_id"].map(abbrev_map)
    elo["game_date"] = pd.to_datetime(elo["game_date"])

    # Rank all teams by their final elo_pre and keep only the top N.
    final_elo_by_team = (
        elo.dropna(subset=["elo_pre"])
        .sort_values("game_date")
        .groupby("team_abbreviation")["elo_pre"]
        .last()
        .sort_values(ascending=False)
    )
    top_teams = final_elo_by_team.index[:top_n].tolist()
    elo = elo[elo["team_abbreviation"].isin(top_teams)]

    fig, ax = styled_subplots((13, 6))

    final_elos: dict[str, tuple] = {}
    for team in top_teams:
        data = elo[elo["team_abbreviation"] == team].sort_values("game_date")
        color = NBA_TEAM_COLORS.get(team, BLUE)
        ax.plot(
            data["game_date"], data["elo_pre"],
            linewidth=1.8, alpha=0.9,
            color=color,
            label=team,
            zorder=2,
        )
        if not data.empty:
            final_elos[team] = (data["game_date"].iloc[-1], data["elo_pre"].iloc[-1])

    # Label every plotted team at the end of its line.
    label_offset = pd.Timedelta(days=3)
    for team in top_teams:
        x, y = final_elos[team]
        ax.text(
            x + label_offset, y, team,
            color=NBA_TEAM_COLORS.get(team, BLUE),
            fontsize=8, fontweight="bold",
            va="center", ha="left",
            clip_on=False,
            zorder=4,
        )

    # Extend x-axis right margin to fit end labels.
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + 18)

    ax.axhline(1500, color=TEXT, linewidth=0.7, linestyle="--", alpha=0.4, zorder=1)

    locator   = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.label.set_color(TEXT)

    ax.set_xlabel("Date", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Elo rating", color=TEXT, fontsize=9, labelpad=8)

    ax.legend(
        frameon=True, facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=8,
        ncol=2, loc="upper left",
        bbox_to_anchor=(0.01, 0.99), borderaxespad=0,
    )

    fig.suptitle(
        f"Elo ratings — top {top_n} teams — {season_api(SEASON)} NBA regular season",
        color=TEXT, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout()

    save_fig(fig, out_path)
    log.info("Elo top-%d teams → %s", top_n, out_path.relative_to(PROJECT_ROOT))


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

    fig, ax = styled_subplots((13, 5))

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
    legend(ax, loc="lower right")
    fig.suptitle("Prediction accuracy throughout the season", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    save_fig(fig, out_path)
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

    fig, ax = styled_subplots((6, 6))

    ax.plot(fpr, tpr, color=BLUE, linewidth=1.8, label=f"AUC = {auc:.4f}", zorder=3)
    ax.plot([0, 1], [0, 1], color=TEXT, linewidth=1, linestyle="--", label="Random classifier", zorder=2)
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE, zorder=1)

    ax.set_xlabel("False positive rate", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("True positive rate", color=TEXT, fontsize=9, labelpad=8)
    legend(ax, loc="lower right")
    fig.suptitle("ROC curve", color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    save_fig(fig, out_path)
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

    fig, ax = styled_subplots((8, 4))

    ax.hist(wins,   bins=bins, alpha=0.55, color=BLUE, label="Home win",  zorder=2)
    ax.hist(losses, bins=bins, alpha=0.55, color=RED,  label="Home loss", zorder=2)
    ax.axvline(0.5, color=TEXT, linewidth=0.9, linestyle="--", zorder=3)

    ax.set_xlabel("Predicted home-win probability", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Games", color=TEXT, fontsize=9, labelpad=8)
    legend(ax)
    fig.suptitle(
        "Distribution of predicted probabilities by actual outcome",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout()

    save_fig(fig, out_path)
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

    fig, ax1 = styled_subplots((10, 4))

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

    save_fig(fig, out_path)
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

    fig, ax = styled_subplots((14, 5))

    ax.bar(team_acc["team"], team_acc["accuracy"], color=colors, zorder=2)
    ax.axhline(overall, color=TEXT, linewidth=1, linestyle="--",
               label=f"Overall mean  {overall:.3f}", zorder=3)

    ax.set_xlabel("Team", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylabel("Accuracy", color=TEXT, fontsize=9, labelpad=8)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    legend(ax)
    fig.suptitle(
        "Per-team prediction accuracy (home + away games combined)",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout()

    save_fig(fig, out_path)
    log.info("Team accuracy → %s", out_path.relative_to(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained NBA win-probability model.")
    p.add_argument(
        "--model",
        choices=["logreg", "xgboost"],
        default="logreg",
        help="Model type to evaluate (default: logreg).",
    )
    p.add_argument(
        "--output-json",
        action="store_true",
        default=False,
        help="Write metrics to outputs/models/latest_metrics.json.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    model_name = args.model

    model_path = MODELS_DIR / f"win_probability_{model_name}.joblib"
    pred_path  = MODELS_DIR / f"rolling_predictions_{model_name}.parquet"
    fig_dir    = FIGURES_LOGREG_DIR if model_name == "logreg" else FIGURES_XGBOOST_DIR

    log.info("Loading model from %s ...", model_path.relative_to(PROJECT_ROOT))
    model = joblib.load(model_path)

    log.info("Loading rolling predictions from %s ...", pred_path.relative_to(PROJECT_ROOT))
    preds = read_parquet(pred_path)

    log.info("Loading processed features ...")
    team_features = read_processed("team_features.parquet")
    elo_ratings   = read_processed("elo_ratings.parquet")

    y_true  = preds["home_win"]
    y_proba = preds["predicted_proba"].values
    y_pred  = preds["predicted_label"].values

    print_metrics(y_true, y_proba, y_pred)

    if args.output_json:
        metrics = {
            "date": date.today().isoformat(),
            "model": model_name,
            "log_loss": float(log_loss(y_true, y_proba)),
            "brier": float(brier_score_loss(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }
        out_path = MODELS_DIR / "latest_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        log.info("Metrics JSON → %s", out_path.relative_to(PROJECT_ROOT))

    if model_name == "logreg":
        plot_feature_coefficients(model, out_path=fig_dir / "feature_coefficients.png")
    else:
        plot_feature_importance(model, out_path=fig_dir / "feature_importance.png")

    plot_elo_time_series(elo_ratings, team_features)
    plot_elo_all_teams(elo_ratings, team_features)
    plot_calibration_curve(y_true, y_proba, out_path=fig_dir / "calibration_curve.png")
    plot_rolling_accuracy(preds, out_path=fig_dir / "rolling_accuracy.png")
    plot_roc_curve(y_true, y_proba, out_path=fig_dir / "roc_curve.png")
    plot_confidence_histogram(y_true, y_proba, out_path=fig_dir / "confidence_histogram.png")
    plot_accuracy_by_confidence(y_true, y_proba, out_path=fig_dir / "accuracy_by_confidence.png")
    plot_team_accuracy(preds, team_features, out_path=fig_dir / "team_accuracy.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
