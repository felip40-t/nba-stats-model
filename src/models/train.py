"""
train.py — Rolling day-by-day win-probability model.

Simulation
----------
Rather than a static train/test split, this script replays the season
chronologically.  For each game date ``d``:

1. Features are recomputed from interim data for all games up to and
   including ``d``.  Because rolling stats use ``shift(1)``, the feature
   row for a game on date ``d`` reflects only games strictly before ``d`` —
   the correct entering-game state, with zero leakage.
2. A fresh logistic regression is fitted on all completed games before ``d``.
3. The model predicts win probability for every game on date ``d``.

The resulting ``rolling_predictions.parquet`` covers the entire season —
each game predicted exactly once using only information available before
it was played.  This mirrors real deployment (predict today's games, update
with results, retrain for tomorrow).

At the end a final model is trained on the full season and saved alongside
the predictions.

Feature set
-----------
elo_delta         home_elo_pre - away_elo_pre
home_adv          home team's Elo home-court bonus
win_rate_delta    home_season_win_rate - away_season_win_rate
pts_delta         home_season_avg_pts - away_season_avg_pts
fg_pct_delta      home_season_avg_fg_pct - away_season_avg_fg_pct
fatigue_delta     home_team_fatigue - away_team_fatigue
acwr_delta        home_team_acwr - away_team_acwr

Run from the project root::

    python src/models/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import compute_features_from_data  # noqa: E402
from src.utils.io import MODELS_DIR, read_interim, read_schedule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_TEAM_FEATURE_COLS = [
    "elo_pre",
    "season_win_rate",
    "season_avg_pts",
    "season_avg_fg_pct",
    "team_fatigue",
    "team_acwr",
]

MODEL_FEATURES = [
    "elo_delta",
    "home_adv",
    "win_rate_delta",
    "pts_delta",
    "fg_pct_delta",
    "fatigue_delta",
    "acwr_delta",
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

    games = (
        sched
        .merge(home_tf, on=["game_id", "home_team_id"], how="inner")
        .merge(away_tf, on=["game_id", "away_team_id"], how="inner")
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
    games["fatigue_delta"] = games["home_team_fatigue"] - games["away_team_fatigue"]
    games["acwr_delta"] = games["home_team_acwr"] - games["away_team_acwr"]
    return games


def drop_missing(games: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any model feature is NaN."""
    before = len(games)
    games = games.dropna(subset=MODEL_FEATURES + ["home_win"]).copy()
    dropped = before - len(games)
    if dropped:
        log.debug("Dropped %d game(s) with NaN features.", dropped)
    return games


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(train: pd.DataFrame) -> Pipeline:
    """Fit a logistic regression pipeline on the given game rows."""
    X_train = train[MODEL_FEATURES]
    y_train = train["home_win"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Rolling simulation
# ---------------------------------------------------------------------------

def build_rolling_predictions(
    game_log: pd.DataFrame,
    player_game_log: pd.DataFrame,
    team_advanced: pd.DataFrame,
    schedule: pd.DataFrame,
    min_train_games: int = 50,
) -> tuple[pd.DataFrame, Pipeline]:
    """Simulate the model predicting each game day using only prior data.

    Iterates over every unique game date ``d`` in the season.  For each date:
    - Computes features for all games up to and including ``d`` (shift-1
      ensures game rows on ``d`` carry only pre-``d`` information).
    - Trains on all complete game rows with ``game_date < d``.
    - Predicts all games on ``d``.

    Skips dates where the training set has fewer than ``min_train_games``
    complete rows (early-season cold start).

    Returns
    -------
    rolling_predictions : pd.DataFrame
        One row per predicted game with columns ``game_id``, ``game_date``,
        ``home_team_id``, ``away_team_id``, ``home_win``,
        ``predicted_proba``, ``predicted_label``.
    final_model : Pipeline
        Model retrained on the full season's data.
    """
    game_dates = sorted(pd.to_datetime(game_log["game_date"]).dt.date.unique())
    all_predictions: list[pd.DataFrame] = []

    for i, date in enumerate(game_dates):
        cutoff = pd.Timestamp(date)

        result = compute_features_from_data(
            game_log, player_game_log, team_advanced, cutoff_date=cutoff
        )
        games = build_game_rows(result["team_features"], schedule)
        games = compute_deltas(games)
        games = drop_missing(games)

        train = games[games["game_date"].dt.date < date]
        today = games[games["game_date"].dt.date == date]

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

    # Final model trained on the full season.
    log.info("Training final model on full season ...")
    full = compute_features_from_data(game_log, player_game_log, team_advanced)
    all_games = build_game_rows(full["team_features"], schedule)
    all_games = compute_deltas(all_games)
    all_games = drop_missing(all_games)
    final_model = train_model(all_games)
    log.info("  final training set : %d games", len(all_games))

    combined = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    return combined, final_model


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading interim data ...")
    game_log = read_interim("game_log.parquet")
    player_game_log = read_interim("player_game_log.parquet")
    team_advanced = read_interim("team_advanced.parquet")
    schedule = read_schedule()
    log.info("  game_log : %d rows", len(game_log))
    log.info("  schedule : %d games", len(schedule))

    log.info("Starting rolling day-by-day simulation ...")
    rolling_preds, final_model = build_rolling_predictions(
        game_log, player_game_log, team_advanced, schedule
    )
    log.info("  predictions collected : %d", len(rolling_preds))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "win_probability_logreg.joblib"
    joblib.dump(final_model, model_path)
    log.info("Model saved → %s", model_path.relative_to(PROJECT_ROOT))

    pred_path = MODELS_DIR / "rolling_predictions.parquet"
    rolling_preds.to_parquet(pred_path, index=False)
    log.info("Rolling predictions saved → %s", pred_path.relative_to(PROJECT_ROOT))

    print("\nDone.")


if __name__ == "__main__":
    main()
