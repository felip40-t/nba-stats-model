"""Smoke tests for src/models/train.py."""

from __future__ import annotations

import pandas as pd

from src.models.train import MODEL_FEATURES, compute_deltas, drop_missing


def _game_row(**overrides):
    row = {
        "home_elo_pre": 1600, "away_elo_pre": 1500,
        "home_season_win_rate": 0.6, "away_season_win_rate": 0.45,
        "home_season_avg_pts": 115, "away_season_avg_pts": 110,
        "home_season_avg_fg_pct": 0.48, "away_season_avg_fg_pct": 0.46,
        "home_team_fatigue": 30.0, "away_team_fatigue": 32.0,
        "home_team_acwr": 1.1, "away_team_acwr": 1.05,
        "home_season_avg_ast": 26, "away_season_avg_ast": 24,
        "home_season_avg_reb": 44, "away_season_avg_reb": 42,
        "home_season_avg_oreb": 10, "away_season_avg_oreb": 9,
        "home_season_avg_blk": 5, "away_season_avg_blk": 4,
        "home_season_avg_stl": 8, "away_season_avg_stl": 7,
        "home_season_avg_tov": 13, "away_season_avg_tov": 14,
        "home_season_avg_pf": 20, "away_season_avg_pf": 21,
        "home_season_avg_fta": 22, "away_season_avg_fta": 20,
        "home_season_avg_ft_pct": 0.78, "away_season_avg_ft_pct": 0.76,
        "home_season_avg_plus_minus": 3.0, "away_season_avg_plus_minus": -1.0,
        "home_adv": 110.0, "home_win": 1,
    }
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
