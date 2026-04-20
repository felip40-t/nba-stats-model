"""Smoke tests for src/models/train.py."""

from __future__ import annotations

import pandas as pd

from src.models.train import MODEL_FEATURES, _LAST10_STATS, compute_deltas, drop_missing

_SEASON_AVG_BASE = {
    "pts": (115, 110), "reb": (44, 42), "oreb": (10, 9), "ast": (26, 24),
    "stl": (8, 7), "blk": (5, 4), "tov": (13, 14), "pf": (20, 21),
    "fta": (22, 20), "fg_pct": (0.48, 0.46), "fg3_pct": (0.36, 0.35),
    "ft_pct": (0.78, 0.76), "plus_minus": (3.0, -1.0),
    "true_shooting_pct": (0.58, 0.56), "three_point_rate": (0.38, 0.36),
    "free_throw_rate": (0.22, 0.20), "oreb_pct_proxy": (0.26, 0.24),
    "true_oreb_pct": (0.28, 0.26),
    "opp_pts": (108, 112), "opp_oreb": (9, 11), "opp_oreb_pct": (0.24, 0.27),
    "opp_blk": (4, 5), "opp_stl": (7, 8),
}


def _game_row(**overrides):
    row = {
        "home_elo_pre": 1600, "away_elo_pre": 1500,
        "home_season_win_rate": 0.6, "away_season_win_rate": 0.45,
        "home_team_fatigue": 30.0, "away_team_fatigue": 32.0,
        "home_team_acwr": 1.1, "away_team_acwr": 1.05,
        "home_recent_form_5": 0.6, "away_recent_form_5": 0.4,
        "home_recent_form_10": 0.55, "away_recent_form_10": 0.45,
        "home_recent_form_15": 0.53, "away_recent_form_15": 0.47,
        "home_win_streak": 2, "away_win_streak": -1,
        "home_h2h_win_rate": 0.6, "away_h2h_win_rate": 0.4,
        "home_adv": 110.0, "home_win": 1,
    }
    for stat, (h_val, a_val) in _SEASON_AVG_BASE.items():
        row[f"home_season_avg_{stat}"] = h_val
        row[f"away_season_avg_{stat}"] = a_val
    for col in _LAST10_STATS:
        row[f"home_last10_avg_{col}"] = _SEASON_AVG_BASE[col][0]
        row[f"away_last10_avg_{col}"] = _SEASON_AVG_BASE[col][1]
    row.update(overrides)
    return row


def test_compute_deltas_produces_expected_signs():
    games = pd.DataFrame([_game_row()])
    out = compute_deltas(games)
    assert out["elo_delta"].iloc[0] == 100
    assert out["pts_delta"].iloc[0] == 5
    assert out["win_rate_delta"].iloc[0] == 0.6 - 0.45
    # All delta columns present
    for col in MODEL_FEATURES:
        assert col in out.columns


def test_drop_missing_removes_rows_with_nan_features():
    good = _game_row()
    bad = _game_row(home_elo_pre=float("nan"))
    games = compute_deltas(pd.DataFrame([good, bad]))
    out = drop_missing(games)
    assert len(out) == 1
