"""
xgboost_grid_search.py  —  Grid search over XGBoost hyperparameters to minimise log-loss.

Evaluation uses a single chronological holdout split: the first TRAIN_FRACTION of games
(by game_date) form the training set; the remaining games form the test set.  This mirrors
the no-lookahead guarantee already baked into the processed features (shift(1).expanding()),
while keeping each combo to a single training pass.

Run from the project root::

    python src/models/xgboost_grid_search.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.display import print_table
from src.utils.io import OUTPUTS_DIR, PROJECT_ROOT, configure_logging, read_processed, read_schedule, write_parquet
from src.models.train import (
    XGBOOST_MODEL_FEATURES,
    build_game_rows,
    compute_deltas,
    drop_missing,
)

log = configure_logging("xgboost_grid_search")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fraction of games (chronological) used for training.  The rest form the
# holdout set used to evaluate each parameter combination.
TRAIN_FRACTION: float = 0.65

# Fixed XGBoost settings not included in the grid.
FIXED_PARAMS: dict = {
    "objective":    "binary:logistic",
    "eval_metric":  "logloss",
    "random_state": 42,
    "n_jobs":       -1,
    "verbosity":    0,
}

# Parameters to search.  Each key maps to a list of candidate values.
# Total combinations = product of all list lengths.
GRID: dict[str, list] = {
    "n_estimators":     [100, 200, 300, 400],
    "max_depth":        [2, 3, 4],
    "learning_rate":    [0.01, 0.02, 0.03],
    "subsample":        [0.7, 0.8],
    "colsample_bytree": [0.7],
    "min_child_weight": [4, 5, 6],
    "gamma":            [0.0, 0.1, 0.2],
    "reg_alpha":        [0.0, 0.1],
    "reg_lambda":       [0.75, 1.0, 1.25],
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    games: pd.DataFrame,
    params: dict,
    train_fraction: float = TRAIN_FRACTION,
) -> float:
    """Train XGBoost on the first ``train_fraction`` of games, return holdout log-loss."""
    games_sorted = games.sort_values("game_date").reset_index(drop=True)
    split = int(len(games_sorted) * train_fraction)

    train = games_sorted.iloc[:split]
    test  = games_sorted.iloc[split:]

    if len(train) == 0 or len(test) == 0:
        return float("nan")

    X_train = train[XGBOOST_MODEL_FEATURES]
    y_train = train["home_win"]
    X_test  = test[XGBOOST_MODEL_FEATURES]
    y_test  = test["home_win"]

    model = XGBClassifier(**params, **FIXED_PARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    proba = np.clip(proba, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y_test, proba))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading processed team features ...")
    team_features = read_processed("team_features.parquet")
    schedule = read_schedule()
    log.info("  team_features : %d rows", len(team_features))
    log.info("  schedule      : %d games", len(schedule))

    log.info("Building game rows and computing deltas ...")
    games = build_game_rows(team_features, schedule)
    games = compute_deltas(games)
    games = drop_missing(games, features=XGBOOST_MODEL_FEATURES)
    log.info("  complete game rows : %d", len(games))

    split_idx = int(len(games.sort_values("game_date")) * TRAIN_FRACTION)
    log.info(
        "Holdout split: train=%d games, test=%d games (TRAIN_FRACTION=%.2f)",
        split_idx, len(games) - split_idx, TRAIN_FRACTION,
    )
    log.info("Model features (%d): %s", len(XGBOOST_MODEL_FEATURES), XGBOOST_MODEL_FEATURES)

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)
    log.info("Starting grid search: %d combinations", total)

    results: list[dict] = []
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        try:
            ll = _evaluate(games, params)
        except Exception:
            log.exception("Combination %d/%d failed: params=%s", idx, total, params)
            ll = float("nan")
        results.append({**params, "log_loss": ll})

        if idx % 50 == 0 or idx == total:
            valid = [r["log_loss"] for r in results if not np.isnan(r["log_loss"])]
            best  = f"{min(valid):.5f}" if valid else "n/a"
            log.info("%6d / %d  |  latest=%.5f  best=%s", idx, total, ll, best)

    n_failed = sum(np.isnan(r["log_loss"]) for r in results)
    if n_failed:
        log.warning("%d / %d combinations returned NaN log-loss", n_failed, total)

    results_df = (
        pd.DataFrame(results)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )

    out_path = OUTPUTS_DIR / "models/xgboost_grid_search_results.parquet"
    write_parquet(results_df, out_path)
    log.info("Saved %d results → %s", len(results_df), out_path.relative_to(PROJECT_ROOT))

    print_table("Top 10 XGBoost parameter combinations", results_df.head(10))


if __name__ == "__main__":
    main()
