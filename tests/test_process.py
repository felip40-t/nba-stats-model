"""Smoke tests for src/data/process.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.process import (
    _add_is_back_to_back,
    _compute_num_ot,
    _convert_minutes,
    _snake,
    build_team_advanced,
)


def test_snake_handles_camel_and_upper():
    assert _snake("teamId") == "team_id"
    assert _snake("GAME_DATE") == "game_date"
    assert _snake("personId") == "person_id"
    assert _snake("FG3_PCT") == "fg3_pct"


def test_convert_minutes():
    assert _convert_minutes("32:30") == 32 + 30 / 60
    assert _convert_minutes("0:00") == 0.0
    assert np.isnan(_convert_minutes(""))
    assert np.isnan(_convert_minutes(None))
    assert np.isnan(_convert_minutes("bad"))


def test_add_is_back_to_back_preserves_row_order():
    df = pd.DataFrame(
        {
            "team_id": [1, 2, 1, 2],
            "game_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-05"]
            ),
        }
    )
    original_index = df.index.tolist()
    out = _add_is_back_to_back(df)
    # Index/order must match the input.
    assert out.index.tolist() == original_index
    # Team 1 played Jan 1 then Jan 2 → back-to-back on the Jan-2 row.
    b2b = out.set_index(["team_id", "game_date"])["is_back_to_back"]
    assert b2b.loc[(1, pd.Timestamp("2025-01-02"))] is np.True_ or b2b.loc[(1, pd.Timestamp("2025-01-02"))]
    # Team 2 Jan 2 is its first game → not back-to-back.
    assert not b2b.loc[(2, pd.Timestamp("2025-01-02"))]


def test_compute_num_ot_zero_and_one():
    # Regulation: 5 players × 48 = 240 player-minutes per team.
    # 1 OT: 240 + 25 = 265 per team.
    rows = []
    for team_id in (10, 11):
        for p in range(5):
            rows.append(
                {
                    "game_id": "G1",
                    "team_id": team_id,
                    "person_id": team_id * 10 + p,
                    "minutes_decimal": 48.0,
                }
            )
        for p in range(5):
            rows.append(
                {
                    "game_id": "G2",
                    "team_id": team_id,
                    "person_id": team_id * 10 + p,
                    "minutes_decimal": 53.0,  # 5 × 53 = 265 → 1 OT
                }
            )
    plog = pd.DataFrame(rows)
    out = _compute_num_ot(plog).set_index("game_id")["num_ot"]
    assert out.loc["G1"] == 0
    assert out.loc["G2"] == 1


def test_build_team_advanced_derived_metrics():
    df = pd.DataFrame(
        [
            # team 1: starters row
            {
                "game_id": "G1", "team_id": 1, "team_city": "A", "team_name": "A",
                "team_tricode": "AAA", "stat_type": "team",
                "fieldGoalsMade": 30, "fieldGoalsAttempted": 60,
                "threePointersMade": 10, "threePointersAttempted": 25,
                "freeThrowsMade": 15, "freeThrowsAttempted": 20,
                "reboundsOffensive": 10, "reboundsDefensive": 30, "reboundsTotal": 40,
                "assists": 20, "steals": 5, "blocks": 3, "turnovers": 12,
                "foulsPersonal": 15, "points": 85,
            },
        ]
    )
    out = build_team_advanced(df)
    row = out.iloc[0]
    # TS% = 85 / (2 × (60 + 0.44 × 20)) = 85 / (2 × 68.8) = 0.6177...
    assert row["true_shooting_pct"] == 85 / (2 * (60 + 0.44 * 20))
    assert row["three_point_rate"] == 25 / 60
    assert row["free_throw_rate"] == 20 / 60
    assert row["oreb_pct_proxy"] == 10 / 40
