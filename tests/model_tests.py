"Implementation for testing feature importance and collinearity in the trained model. This is not a unit test but rather a script to run various analyses on the model's features after training."


from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from src.models.train import MODEL_FEATURES, build_game_rows, compute_deltas, drop_missing
from src.utils.io import read_processed, read_schedule
from tests.feature_tests import run_all_feature_tests

# Load artifacts from the last training run
model = joblib.load("outputs/models/win_probability_logreg.joblib")

# Reconstruct the full game-row table with all delta features (rolling_predictions.parquet
# only stores predictions, not the feature columns used to produce them)
team_features = read_processed("team_features.parquet")
schedule = read_schedule()
games = build_game_rows(team_features, schedule)
games = compute_deltas(games)
games = drop_missing(games)

X = games[MODEL_FEATURES]
y = games["home_win"].astype(int)

results = run_all_feature_tests(model, X, y)

# Which features survive medium L1 regularisation?
l1 = pd.DataFrame([vars(f) for f in results["l1_sweep"].features])
print(l1.sort_values("score"))

# Most important features by permutation
perm = pd.DataFrame([vars(f) for f in results["permutation_importance"].features])
print(perm.sort_values("score", ascending=False).head(10))

# Features whose removal hurts performance most
abl = pd.DataFrame([vars(f) for f in results["ablation"].features])
print(abl.sort_values("score", ascending=False).head(10))

# Collinear features (VIF >= 5)
vif = pd.DataFrame([vars(f) for f in results["vif"].features])
print(vif[~vif["passed"]].sort_values("score", ascending=False))
