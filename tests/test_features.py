"""Smoke tests for src/data/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import (
    FATIGUE_LAMBDA,
    _acwr_player,
    _fatigue_decay_player,
    compute_elo_ratings,
)


def test_fatigue_decay_zero_when_single_game():
    g = pd.DataFrame(
        {"game_date": pd.to_datetime(["2025-01-01"]), "minutes_decimal": [32.0]}
    )
    out = _fatigue_decay_player(g, FATIGUE_LAMBDA)
    assert len(out) == 1
    assert out.iloc[0] == 0.0


def test_fatigue_decay_matches_naive_on_small_sequence():
    dates = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-06", "2025-01-10"])
    minutes = np.array([30.0, 28.0, 0.0, 35.0])
    g = pd.DataFrame({"game_date": dates, "minutes_decimal": minutes})

    # Naive O(n²) reference: fatigue_i = Σ_{j<i} m_j · e^{-λ(t_i − t_j)}
    lam = FATIGUE_LAMBDA
    days = (dates - dates[0]).days.to_numpy()
    expected = np.zeros(len(minutes))
    for i in range(1, len(minutes)):
        expected[i] = sum(
            minutes[j] * np.exp(-lam * (days[i] - days[j])) for j in range(i)
        )

    got = _fatigue_decay_player(g, lam).to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_acwr_preserves_index_and_handles_duplicate_dates():
    # Two "games" on the same date — exercise the duplicate-date path.
    g = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-10"]
            ),
            "minutes_decimal": [30.0, 25.0, 20.0, 28.0],
        }
    )
    out = _acwr_player(g)
    assert list(out.index) == list(g.index)
    # First row has no prior history → NaN
    assert np.isnan(out.iloc[0])


def test_compute_elo_sum_conservation():
    # Elo ratings should sum to (n_teams × initial) after each game because
    # home's gain equals away's loss (zero-sum update).
    dates = pd.to_datetime(["2025-01-01", "2025-01-03"])
    game_log = pd.DataFrame(
        [
            {"game_id": "G1", "game_date": dates[0], "team_id": 1,
             "is_home": True, "win": True, "plus_minus": 10, "num_ot": 0},
            {"game_id": "G1", "game_date": dates[0], "team_id": 2,
             "is_home": False, "win": False, "plus_minus": -10, "num_ot": 0},
            {"game_id": "G2", "game_date": dates[1], "team_id": 1,
             "is_home": False, "win": False, "plus_minus": -5, "num_ot": 0},
            {"game_id": "G2", "game_date": dates[1], "team_id": 2,
             "is_home": True, "win": True, "plus_minus": 5, "num_ot": 0},
        ]
    )
    elo = compute_elo_ratings(game_log)
    # Two rows per game.
    assert len(elo) == 4
    # Post totals must equal the initial sum (zero-sum).
    total_post = elo.groupby("game_id")["elo_post"].sum()
    assert np.allclose(total_post.values, 2 * 1500.0)
    # Winner gains, loser loses.
    g1 = elo[elo["game_id"] == "G1"]
    winner = g1[g1["team_id"] == 1].iloc[0]
    loser = g1[g1["team_id"] == 2].iloc[0]
    assert winner["elo_post"] > winner["elo_pre"]
    assert loser["elo_post"] < loser["elo_pre"]


def test_compute_elo_no_leakage_pre_vs_post():
    """elo_pre must be the rating going INTO the game, not the updated one."""
    dates = pd.to_datetime(["2025-01-01", "2025-01-03"])
    game_log = pd.DataFrame(
        [
            {"game_id": "G1", "game_date": dates[0], "team_id": 1,
             "is_home": True, "win": True, "plus_minus": 10, "num_ot": 0},
            {"game_id": "G1", "game_date": dates[0], "team_id": 2,
             "is_home": False, "win": False, "plus_minus": -10, "num_ot": 0},
            {"game_id": "G2", "game_date": dates[1], "team_id": 1,
             "is_home": False, "win": False, "plus_minus": -5, "num_ot": 0},
            {"game_id": "G2", "game_date": dates[1], "team_id": 2,
             "is_home": True, "win": True, "plus_minus": 5, "num_ot": 0},
        ]
    )
    elo = compute_elo_ratings(game_log)
    # Team 1: elo_post of G1 == elo_pre of G2
    t1 = elo[elo["team_id"] == 1].sort_values("game_id")
    assert t1.iloc[0]["elo_post"] == t1.iloc[1]["elo_pre"]
