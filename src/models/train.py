"""
train.py — Win-probability model training.

Two modes
---------
Rolling simulation (default)
    Iterates over every game date ``d`` in the season.  For each date:
    1. Trains on all complete game rows with ``game_date < d``.
    2. Predicts all games on ``d``.
    Features are computed once upfront from the processed snapshot on disk
    (written by features.py) — no per-iteration recompute.  Because the
    feature table uses ``shift(1).expanding()``, each game row already carries
    only pre-game information, so slicing by date is sufficient and safe.

Playoffs mode (``--playoffs``)
    Trains a single model on the entire regular-season processed feature
    snapshot and saves it as ``win_probability_logreg_playoffs.joblib``.
    Use this after the regular season ends to prepare a model ready to
    predict playoff match-ups.

Feature set
-----------
elo_delta           home_elo_pre - away_elo_pre
home_adv            home team's Elo home-court bonus
win_rate_delta      home_season_win_rate - away_season_win_rate
pts_delta           home_season_avg_pts - away_season_avg_pts
fg_pct_delta            home_season_avg_fg_pct - away_season_avg_fg_pct
fg3_pct_delta           home_season_avg_fg3_pct - away_season_avg_fg3_pct
fatigue_delta           home_team_fatigue - away_team_fatigue
acwr_delta              home_team_acwr - away_team_acwr
ast_delta               home_season_avg_ast - away_season_avg_ast
reb_delta               home_season_avg_reb - away_season_avg_reb
oreb_delta              home_season_avg_oreb - away_season_avg_oreb
blk_delta               home_season_avg_blk - away_season_avg_blk
stl_delta               home_season_avg_stl - away_season_avg_stl
tov_delta               home_season_avg_tov - away_season_avg_tov
pf_delta                home_season_avg_pf - away_season_avg_pf
fta_delta               home_season_avg_fta - away_season_avg_fta
ft_pct_delta            home_season_avg_ft_pct - away_season_avg_ft_pct
true_shooting_pct_delta home_season_avg_true_shooting_pct - away_season_avg_true_shooting_pct
three_point_rate_delta  home_season_avg_three_point_rate - away_season_avg_three_point_rate
free_throw_rate_delta   home_season_avg_free_throw_rate - away_season_avg_free_throw_rate
oreb_pct_proxy_delta    home_season_avg_oreb_pct_proxy - away_season_avg_oreb_pct_proxy
true_oreb_pct_delta     home_season_avg_true_oreb_pct - away_season_avg_true_oreb_pct
opp_pts_delta           home_season_avg_opp_pts - away_season_avg_opp_pts
opp_oreb_delta          home_season_avg_opp_oreb - away_season_avg_opp_oreb
opp_oreb_pct_delta      home_season_avg_opp_oreb_pct - away_season_avg_opp_oreb_pct
opp_blk_delta           home_season_avg_opp_blk - away_season_avg_opp_blk
opp_stl_delta           home_season_avg_opp_stl - away_season_avg_opp_stl
last10_<stat>_delta     home_last10_avg_<stat> - away_last10_avg_<stat>  (23 stats, same set as season_avg_*)
recent_form_5_delta     home_recent_form_5 - away_recent_form_5
recent_form_10_delta    home_recent_form_10 - away_recent_form_10
recent_form_15_delta    home_recent_form_15 - away_recent_form_15
win_streak_delta        home_win_streak - away_win_streak
h2h_delta               home_h2h_win_rate - away_h2h_win_rate

Run from the project root::

    python src/models/train.py                  # rolling simulation
    python src/models/train.py --playoffs        # full-season model for playoffs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_HERE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERE_PROJECT_ROOT))

from src.utils.io import MODELS_DIR, PROJECT_ROOT, read_processed, read_schedule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Stats for which both a season-to-date average and a 10-game rolling average
# are computed in features.py.  Used to auto-generate _TEAM_FEATURE_COLS entries
# and MODEL_FEATURES delta columns without listing them twice.
_LAST10_STATS = [
    "pts", "reb", "oreb", "ast", "stl", "blk", "tov", "pf", "fta",
    "fg_pct", "fg3_pct", "ft_pct", "plus_minus",
    "true_shooting_pct", "three_point_rate", "free_throw_rate", "oreb_pct_proxy",
    "true_oreb_pct",
    "opp_pts", "opp_oreb", "opp_oreb_pct", "opp_blk", "opp_stl",
]

_TEAM_FEATURE_COLS = [
    "elo_pre",
    "season_win_rate",
    "season_avg_pts",
    "season_avg_fg_pct",
    "season_avg_fg3_pct",
    "season_avg_ast",
    "season_avg_reb",
    "season_avg_oreb",
    "season_avg_blk",
    "season_avg_stl",
    "season_avg_tov",
    "season_avg_pf",
    "season_avg_fta",
    "season_avg_ft_pct",
    "season_avg_true_shooting_pct",
    "season_avg_three_point_rate",
    "season_avg_free_throw_rate",
    "season_avg_oreb_pct_proxy",
    "season_avg_true_oreb_pct",
    "season_avg_opp_pts",
    "season_avg_opp_oreb",
    "season_avg_opp_oreb_pct",
    "season_avg_opp_blk",
    "season_avg_opp_stl",
    "recent_form_5",
    "recent_form_10",
    "recent_form_15",
    "win_streak",
    "h2h_win_rate",
    "team_fatigue",
    "team_acwr",
] + [f"last10_avg_{c}" for c in _LAST10_STATS]

MODEL_FEATURES = [
    "elo_delta",
    "home_adv",
    "fatigue_delta",
    "acwr_delta",
    "h2h_delta",
]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_game_rows(
    team_features: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one row per played game with home/away feature columns."""
    tf = team_features.copy()
    tf["team_id"] = tf["team_id"].astype("Int64")

    home_cols = ["game_id", "game_date", "team_id", "win", "home_adv"] + _TEAM_FEATURE_COLS
    home_tf = tf[home_cols].copy()
    home_tf = home_tf.rename(
        columns=(
            {"team_id": "home_team_id", "win": "home_win", "home_adv": "home_adv"}
            | {c: f"home_{c}" for c in _TEAM_FEATURE_COLS}
        )
    )

    away_cols = ["game_id", "team_id"] + _TEAM_FEATURE_COLS
    away_tf = tf[away_cols].copy()
    away_tf = away_tf.rename(
        columns={"team_id": "away_team_id"}
        | {c: f"away_{c}" for c in _TEAM_FEATURE_COLS}
    )

    sched = schedule[["game_id", "home_team_id", "away_team_id"]].copy()
    sched["home_team_id"] = sched["home_team_id"].astype("Int64")
    sched["away_team_id"] = sched["away_team_id"].astype("Int64")

    # Inner-join drops games not represented in team_features.  That is usually
    # fine (unplayed future games, or pre-threshold rows), but it can also mask
    # a data bug (e.g. a team_id mismatch) — warn so callers see the scale.
    games = (
        sched
        .merge(home_tf, on=["game_id", "home_team_id"], how="inner")
        .merge(away_tf, on=["game_id", "away_team_id"], how="inner")
    )
    scheduled = len(sched)
    matched = len(games)
    if matched < scheduled:
        log.info(
            "build_game_rows: matched %d of %d scheduled games to team features (dropped %d).",
            matched, scheduled, scheduled - matched,
        )

    games["home_win"] = games["home_win"].astype(int)
    return games.sort_values("game_date").reset_index(drop=True)


def compute_deltas(games: pd.DataFrame) -> pd.DataFrame:
    """Add model input columns derived from home/away feature differences."""
    games = games.copy()
    games["elo_delta"] = games["home_elo_pre"] - games["away_elo_pre"]
    games["win_rate_delta"] = games["home_season_win_rate"] - games["away_season_win_rate"]
    games["pts_delta"] = games["home_season_avg_pts"] - games["away_season_avg_pts"]
    games["fg_pct_delta"] = games["home_season_avg_fg_pct"] - games["away_season_avg_fg_pct"]
    games["fg3_pct_delta"] = games["home_season_avg_fg3_pct"] - games["away_season_avg_fg3_pct"]
    games["fatigue_delta"] = games["home_team_fatigue"] - games["away_team_fatigue"]
    games["acwr_delta"] = games["home_team_acwr"] - games["away_team_acwr"]
    games["ast_delta"] = games["home_season_avg_ast"] - games["away_season_avg_ast"]
    games["reb_delta"] = games["home_season_avg_reb"] - games["away_season_avg_reb"]
    games["oreb_delta"] = games["home_season_avg_oreb"] - games["away_season_avg_oreb"]
    games["blk_delta"] = games["home_season_avg_blk"] - games["away_season_avg_blk"]
    games["stl_delta"] = games["home_season_avg_stl"] - games["away_season_avg_stl"]
    games["tov_delta"] = games["home_season_avg_tov"] - games["away_season_avg_tov"]
    games["pf_delta"] = games["home_season_avg_pf"] - games["away_season_avg_pf"]
    games["fta_delta"] = games["home_season_avg_fta"] - games["away_season_avg_fta"]
    games["ft_pct_delta"] = games["home_season_avg_ft_pct"] - games["away_season_avg_ft_pct"]
    games["true_shooting_pct_delta"] = games["home_season_avg_true_shooting_pct"] - games["away_season_avg_true_shooting_pct"]
    games["three_point_rate_delta"] = games["home_season_avg_three_point_rate"] - games["away_season_avg_three_point_rate"]
    games["free_throw_rate_delta"] = games["home_season_avg_free_throw_rate"] - games["away_season_avg_free_throw_rate"]
    games["oreb_pct_proxy_delta"] = games["home_season_avg_oreb_pct_proxy"] - games["away_season_avg_oreb_pct_proxy"]
    games["true_oreb_pct_delta"] = games["home_season_avg_true_oreb_pct"] - games["away_season_avg_true_oreb_pct"]
    games["opp_pts_delta"] = games["home_season_avg_opp_pts"] - games["away_season_avg_opp_pts"]
    games["opp_oreb_delta"] = games["home_season_avg_opp_oreb"] - games["away_season_avg_opp_oreb"]
    games["opp_oreb_pct_delta"] = games["home_season_avg_opp_oreb_pct"] - games["away_season_avg_opp_oreb_pct"]
    games["opp_blk_delta"] = games["home_season_avg_opp_blk"] - games["away_season_avg_opp_blk"]
    games["opp_stl_delta"] = games["home_season_avg_opp_stl"] - games["away_season_avg_opp_stl"]
    games["recent_form_5_delta"] = games["home_recent_form_5"] - games["away_recent_form_5"]
    games["recent_form_10_delta"] = games["home_recent_form_10"] - games["away_recent_form_10"]
    games["recent_form_15_delta"] = games["home_recent_form_15"] - games["away_recent_form_15"]
    games["win_streak_delta"] = games["home_win_streak"] - games["away_win_streak"]
    games["h2h_delta"] = games["home_h2h_win_rate"] - games["away_h2h_win_rate"]
    for col in _LAST10_STATS:
        games[f"last10_{col}_delta"] = games[f"home_last10_avg_{col}"] - games[f"away_last10_avg_{col}"]
    return games


def drop_missing(games: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any model feature is NaN.

    Logs at INFO rather than DEBUG so a silent data bug is immediately visible
    in the training logs.  Early-season drops are expected (features need
    several games of history); an unexpectedly large drop later in the season
    usually means upstream data is missing.
    """
    before = len(games)
    nan_mask = games[MODEL_FEATURES + ["home_win"]].isna().any(axis=1)
    dropped = nan_mask.sum()
    if dropped:
        nan_counts = (
            games.loc[nan_mask, MODEL_FEATURES]
            .isna().sum()
            .pipe(lambda s: s[s > 0])
            .sort_values(ascending=False)
        )
        log.info(
            "drop_missing: dropped %d of %d game rows with NaN features.\n"
            "  NaN counts per feature (top culprits):\n%s",
            dropped, before,
            "\n".join(f"    {col}: {n}" for col, n in nan_counts.items()),
        )
    return games[~nan_mask].copy()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(train: pd.DataFrame) -> Pipeline:
    """Fit a logistic regression pipeline on the given game rows."""
    X_train = train[MODEL_FEATURES]
    y_train = train["home_win"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1500, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Rolling simulation
# ---------------------------------------------------------------------------

def build_rolling_predictions(
    games: pd.DataFrame,
    min_train_games: int = 50,
) -> pd.DataFrame:
    """Simulate the model predicting each game using only prior data.

    Receives the full pre-built, pre-filtered game table (features already
    computed for the whole season upfront).  For each date ``d``:

    - Trains on all complete rows with ``game_date < d``.
    - Predicts all games on ``d``.

    No feature recomputation happens inside this loop — the ``shift(1)``
    encoding in the feature table guarantees that each row already carries
    only pre-game information.

    Skips dates where the training set has fewer than ``min_train_games``
    complete rows (early-season cold start).

    Parameters
    ----------
    games:
        Output of ``drop_missing(compute_deltas(build_game_rows(...)))``.
        One row per played game with all MODEL_FEATURES and ``home_win``.
    min_train_games:
        Minimum training rows required before predictions begin.

    Returns
    -------
    pd.DataFrame
        One row per predicted game with columns ``game_id``, ``game_date``,
        ``home_team_id``, ``away_team_id``, ``home_win``,
        ``predicted_proba``, ``predicted_label``.
    """
    # Work in Timestamp space (faster than Python ``datetime.date`` and avoids
    # per-row object conversion every iteration).  ``normalize()`` zeroes the
    # time-of-day component so same-day games collapse cleanly.
    day = games["game_date"].dt.normalize()
    game_dates = np.sort(day.unique())
    all_predictions: list[pd.DataFrame] = []

    for i, date in enumerate(game_dates):
        train = games[day < date]
        today = games[day == date]

        if len(train) < min_train_games or len(today) == 0:
            log.info("  [%d/%d] %s : skipped (only %d training games)",
                     i + 1, len(game_dates), date, len(train))
            continue

        model = train_model(train)
        proba = model.predict_proba(today[MODEL_FEATURES])[:, 1]
        pred_label = model.predict(today[MODEL_FEATURES])

        pred_df = today[["game_id", "game_date", "home_team_id", "away_team_id", "home_win"]].copy()
        pred_df["predicted_proba"] = proba
        pred_df["predicted_label"] = pred_label
        all_predictions.append(pred_df)

        log.info("  [%d/%d] %s : trained on %d games, predicted %d",
                 i + 1, len(game_dates), date, len(train), len(today))

    return (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the NBA win-probability model.")
    p.add_argument(
        "--playoffs",
        action="store_true",
        help=(
            "Train on all regular-season data at once and save a playoffs-ready "
            "model (win_probability_logreg_playoffs.joblib).  Skips the rolling "
            "simulation."
        ),
    )
    p.add_argument(
        "--min-train-games",
        type=int,
        default=50,
        metavar="N",
        help="Minimum training rows before rolling predictions begin (default: 50).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Load processed features (written by features.py).
    # ------------------------------------------------------------------
    log.info("Loading processed team features ...")
    team_features = read_processed("team_features.parquet")
    schedule = read_schedule()
    log.info("  team_features : %d rows", len(team_features))
    log.info("  schedule      : %d games", len(schedule))

    # ------------------------------------------------------------------
    # Build the full-season game table once — shared by both modes.
    # build_game_rows pivots team features into one row per game;
    # compute_deltas adds the home-minus-away delta columns;
    # drop_missing removes rows where any model feature is NaN (early
    # season games before the cold-start threshold has been reached).
    # ------------------------------------------------------------------
    log.info("Building game rows and computing deltas ...")
    games = build_game_rows(team_features, schedule)
    games = compute_deltas(games)
    games = drop_missing(games)
    log.info("  complete game rows : %d", len(games))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.playoffs:
        # ------------------------------------------------------------------
        # Playoffs mode: train one model on all regular-season data.
        # ------------------------------------------------------------------
        log.info("Playoffs mode: training on full regular-season dataset (%d games) ...", len(games))
        model = train_model(games)

        model_path = MODELS_DIR / "win_probability_logreg_playoffs.joblib"
        joblib.dump(model, model_path)
        log.info("Playoffs model saved → %s", model_path.relative_to(PROJECT_ROOT))

    else:
        # ------------------------------------------------------------------
        # Rolling mode: predict each game using only prior-date data,
        # then train a final model on the complete season.
        # ------------------------------------------------------------------
        log.info("Starting rolling day-by-day simulation ...")
        rolling_preds = build_rolling_predictions(games, min_train_games=args.min_train_games)
        log.info("  predictions collected : %d", len(rolling_preds))

        pred_path = MODELS_DIR / "rolling_predictions.parquet"
        rolling_preds.to_parquet(pred_path, index=False)
        log.info("Rolling predictions saved → %s", pred_path.relative_to(PROJECT_ROOT))

        log.info("Training final model on full season (%d games) ...", len(games))
        final_model = train_model(games)

        model_path = MODELS_DIR / "win_probability_logreg.joblib"
        joblib.dump(final_model, model_path)
        log.info("Model saved → %s", model_path.relative_to(PROJECT_ROOT))

    print("\nDone.")


if __name__ == "__main__":
    main()
