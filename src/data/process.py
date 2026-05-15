"""
process.py — Stage 2 of the NBA stats pipeline.

Reads raw Parquet files from ``data/raw/``, cleans and restructures them
into three interim tables, and writes the results to ``data/interim/``.

Output tables
-------------
game_log.parquet
    One row per team per game — team-level box-score stats with contextual flags.
player_game_log.parquet
    One row per player per game — individual box-score stats.
team_advanced.parquet
    Per-game aggregated team stats derived from the box-score (starters + bench
    combined), including a handful of pace/efficiency metrics.

Run from the project root::

    python src/data/process.py

Playoff data is NOT processed automatically. Pass ``--playoffs`` to read
``*_playoffs.parquet`` raw files and write ``*_playoffs.parquet`` interim
files (e.g. ``game_log_playoffs.parquet``). Fetch playoff raw data first
with ``python src/data/fetch_games.py --playoffs``::

    python src/data/process.py --playoffs
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is on sys.path so ``src`` is importable when
# the script is run directly (python src/data/process.py).
_HERE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_HERE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERE_PROJECT_ROOT))

from src.utils.display import print_table  # noqa: E402
from src.utils.io import PROJECT_ROOT, configure_logging, read_raw, write_interim  # noqa: E402

log = configure_logging("process")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snake(col: str) -> str:
    """Convert a column name to snake_case.

    Handles both ``UPPER_CASE`` (NBA API team-log style) and ``camelCase``
    (NBA API box-score style) by inserting underscores before uppercase
    letters that follow a lowercase letter or digit before lowercasing.
    """
    col = col.strip()
    # Insert underscore before uppercase letters that follow a lowercase letter
    # or digit (camelCase → snake_case).  UPPER_CASE is unaffected because
    # consecutive capitals have no lowercase preceding them.
    col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    return col.lower().replace(" ", "_")


def _convert_minutes(val: str) -> float:
    """Convert a single ``'MM:SS'`` string to decimal minutes, or ``NaN``."""
    if pd.isna(val) or val == "":
        return np.nan
    try:
        parts = str(val).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except (ValueError, IndexError):
        return np.nan


def _parse_minutes(minutes_str: pd.Series) -> pd.Series:
    """Apply ``_convert_minutes`` across a Series of ``'MM:SS'`` strings."""
    return minutes_str.apply(_convert_minutes)


def _coerce_cols(df: pd.DataFrame, cols: list[str], dtype: str | None = None) -> None:
    """Cast columns to dtype in-place via pd.to_numeric, skipping missing columns."""
    for col in cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.astype(dtype) if dtype is not None else s


# ---------------------------------------------------------------------------
# game_log
# ---------------------------------------------------------------------------

def clean_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and restructure the raw team game-log DataFrame.

    Standardises column names, parses dates, casts dtypes, and derives
    three contextual flags: ``is_home``, ``win``, and ``is_back_to_back``.
    ``num_ot`` is added later in ``run_pipeline`` because it requires
    player-minute data from the box-score.

    Parameters
    ----------
    df:
        Raw DataFrame read from ``team_gamelog_raw.parquet``.

    Returns
    -------
    pd.DataFrame
        Cleaned game-log with one row per team per game.
    """
    df = df.copy()

    # --- column names -------------------------------------------------------
    df.columns = [_snake(c) for c in df.columns]

    # --- dtypes -------------------------------------------------------------
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["team_id"] = df["team_id"].astype("Int64")

    _coerce_cols(df, ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                      "oreb", "dreb", "reb", "ast", "stl", "blk",
                      "tov", "pf", "pts", "plus_minus", "min"], "Int64")
    _coerce_cols(df, ["fg_pct", "fg3_pct", "ft_pct"])

    # --- drop uninformative columns -----------------------------------------
    df = df.drop(columns=["video_available"], errors="ignore")

    # --- contextual flags ---------------------------------------------------
    df = _add_is_home(df)
    df = _add_win(df)
    df = _add_is_back_to_back(df)

    # --- tidy column order --------------------------------------------------
    front_cols = ["game_id", "game_date", "season_id", "team_id",
                  "team_abbreviation", "team_name", "matchup",
                  "is_home", "win", "is_back_to_back"]
    remaining = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining]

    return df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


def _add_is_home(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``is_home`` from the ``matchup`` column.

    The API encodes home games as ``'TEAM vs. OPP'`` and away games as
    ``'TEAM @ OPP'``.
    """
    df = df.copy()
    df["is_home"] = df["matchup"].str.contains(r"vs\.", regex=True)
    return df


def _add_win(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a binary ``win`` flag from the ``wl`` column (``'W'``/``'L'``)."""
    df = df.copy()
    df["win"] = df["wl"].str.upper() == "W"
    return df


def _add_is_back_to_back(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``is_back_to_back`` — True when a team played the previous day.

    The flag is computed per team by sorting on ``game_date`` and checking
    whether the gap to the previous game equals exactly one calendar day.
    Input row order is preserved: shifts are computed on a sorted view, then
    the result is reindexed back to the caller's original index.
    """
    df = df.copy()
    sorted_view = df.sort_values(["team_id", "game_date"])
    prev_date = sorted_view.groupby("team_id")["game_date"].shift(1)
    b2b = (sorted_view["game_date"] - prev_date).dt.days == 1
    df["is_back_to_back"] = b2b.reindex(df.index)
    return df


def _compute_num_ot(player_game_log: pd.DataFrame) -> pd.DataFrame:
    """Derive the number of overtime periods per game from summed player minutes.

    NBA regulation = 5 players × 48 min = 240 player-minutes per team.
    Each OT period adds 5 × 5 = 25 player-minutes.  We take the max across
    both teams to guard against foul-outs slightly reducing one team's count.

    Returns a DataFrame with columns ``game_id`` and ``num_ot``.
    """
    team_mins = (
        player_game_log
        .groupby(["game_id", "team_id"])["minutes_decimal"]
        .sum()
        .reset_index(name="total_mins")
    )
    game_mins = team_mins.groupby("game_id")["total_mins"].max().reset_index(name="max_mins")
    game_mins["num_ot"] = (
        ((game_mins["max_mins"] - 240) / 25).clip(lower=0).round().astype("Int64")
    )
    return game_mins[["game_id", "num_ot"]]


# ---------------------------------------------------------------------------
# player_game_log
# ---------------------------------------------------------------------------

def clean_player_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and restructure the player-stats rows from the raw box-score.

    Parameters
    ----------
    df:
        Raw combined box-score DataFrame (both ``stat_type`` values). Player
        rows are isolated internally.

    Returns
    -------
    pd.DataFrame
        Cleaned player game-log with one row per player per game.
    """
    df = df[df["stat_type"] == "player"].copy()

    # --- column names -------------------------------------------------------
    df.columns = [_snake(c) for c in df.columns]

    # --- drop housekeeping columns ------------------------------------------
    df = df.drop(columns=["stat_type", "starters_bench"], errors="ignore")

    # --- dtypes -------------------------------------------------------------
    df["team_id"] = df["team_id"].astype("Int64")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
    df["minutes_decimal"] = _parse_minutes(df["minutes"])

    _coerce_cols(df, ["field_goals_made", "field_goals_attempted",
                      "three_pointers_made", "three_pointers_attempted",
                      "free_throws_made", "free_throws_attempted",
                      "rebounds_offensive", "rebounds_defensive", "rebounds_total",
                      "assists", "steals", "blocks", "turnovers", "fouls_personal", "points"], "Int64")
    _coerce_cols(df, ["field_goals_percentage", "three_pointers_percentage",
                      "free_throws_percentage", "plus_minus_points"])

    # --- tidy column order --------------------------------------------------
    front_cols = ["game_id", "team_id", "team_tricode", "team_city", "team_name",
                  "person_id", "first_name", "family_name", "name_i",
                  "position", "jersey_num", "minutes", "minutes_decimal"]
    remaining = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining]

    return df.sort_values(["game_id", "team_id", "person_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# team_advanced
# ---------------------------------------------------------------------------

def build_team_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-game team totals and basic efficiency metrics from box-score data.

    The box-score stores team stats split into Starters and Bench rows; this
    function aggregates them into a single row per team per game and computes
    a small set of metrics that can be derived without additional data:

    * ``true_shooting_pct``  — TS% = PTS / (2 × (FGA + 0.44 × FTA))
    * ``three_point_rate``   — share of FGA taken from three-point range
    * ``free_throw_rate``    — FTA per FGA
    * ``oreb_pct_proxy``     — team OREB / team total REB (rough proxy)

    Parameters
    ----------
    df:
        Raw combined box-score DataFrame (both ``stat_type`` values). Team
        rows are isolated internally.

    Returns
    -------
    pd.DataFrame
        One row per team per game with aggregated counts and derived metrics.
    """
    team_df = df[df["stat_type"] == "team"].copy()
    team_df.columns = [_snake(c) for c in team_df.columns]

    # Numeric columns to sum across starters + bench
    sum_cols = [
        "field_goals_made", "field_goals_attempted",
        "three_pointers_made", "three_pointers_attempted",
        "free_throws_made", "free_throws_attempted",
        "rebounds_offensive", "rebounds_defensive", "rebounds_total",
        "assists", "steals", "blocks", "turnovers", "fouls_personal", "points",
    ]
    _coerce_cols(team_df, sum_cols)

    group_keys = ["game_id", "team_id", "team_city", "team_name", "team_tricode"]
    agg_df = (
        team_df
        .groupby(group_keys, as_index=False)[sum_cols]
        .sum()
    )

    # --- derived metrics ----------------------------------------------------
    fga = agg_df["field_goals_attempted"]
    fta = agg_df["free_throws_attempted"]
    pts = agg_df["points"]
    oreb = agg_df["rebounds_offensive"]
    reb = agg_df["rebounds_total"]

    agg_df["true_shooting_pct"] = pts / (2 * (fga + 0.44 * fta)).replace(0, np.nan)
    agg_df["three_point_rate"] = (
        agg_df["three_pointers_attempted"] / fga.replace(0, np.nan)
    )
    agg_df["free_throw_rate"] = fta / fga.replace(0, np.nan)
    agg_df["oreb_pct_proxy"] = oreb / reb.replace(0, np.nan)

    # Cast integer columns
    _coerce_cols(agg_df, sum_cols, "Int64")

    return agg_df.sort_values(["game_id", "team_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    playoffs: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute the full processing pipeline and return the three output tables.

    Reads raw files, applies all cleaning and transformation functions, saves
    the results to ``data/interim/``, and (when ``verbose=True``) prints a
    formatted preview of every output table.

    Parameters
    ----------
    playoffs:
        If True, read ``*_playoffs.parquet`` raw files and write
        ``*_playoffs.parquet`` interim files.  Otherwise operate on the
        regular-season files.
    verbose:
        If True, print table previews and file paths to stdout.

    Returns
    -------
    dict with keys ``"game_log"``, ``"player_game_log"``, ``"team_advanced"``.
    """
    suffix = "_playoffs" if playoffs else ""

    if verbose:
        label = "playoffs" if playoffs else "regular-season"
        print(f"Loading raw {label} data...")

    gamelog_raw = read_raw(f"team_gamelog_raw{suffix}.parquet")
    boxscore_raw = read_raw(f"boxscore_raw{suffix}.parquet")

    if verbose:
        print(f"  team_gamelog_raw{suffix} : {gamelog_raw.shape[0]} rows, "
              f"{gamelog_raw.shape[1]} cols")
        print(f"  boxscore_raw{suffix}     : {boxscore_raw.shape[0]} rows, "
              f"{boxscore_raw.shape[1]} cols")

    # --- transform ----------------------------------------------------------
    game_log = clean_game_log(gamelog_raw)
    player_game_log = clean_player_game_log(boxscore_raw)
    team_advanced = build_team_advanced(boxscore_raw)

    # --- add num_ot (requires player minutes from boxscore) -----------------
    num_ot_df = _compute_num_ot(player_game_log)
    game_log = game_log.merge(num_ot_df, on="game_id", how="left")
    # Place num_ot immediately after is_back_to_back
    cols = list(game_log.columns)
    cols.remove("num_ot")
    cols.insert(cols.index("is_back_to_back") + 1, "num_ot")
    game_log = game_log[cols]

    # --- save ---------------------------------------------------------------
    outputs = {
        "game_log": game_log,
        "player_game_log": player_game_log,
        "team_advanced": team_advanced,
    }

    if verbose:
        print("\nSaving to data/interim/ ...")

    for name, df in outputs.items():
        dest = write_interim(df, f"{name}{suffix}.parquet")
        if verbose:
            print(f"  -> {dest.relative_to(PROJECT_ROOT)}")
        log.info("%s%s: %d rows written", name, suffix, len(df))

    # --- display ------------------------------------------------------------
    if verbose:
        print_table(f"game_log{suffix}", game_log)
        print_table(f"player_game_log{suffix}", player_game_log)
        print_table(f"team_advanced{suffix}", team_advanced)

    return outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean raw NBA data into interim tables.")
    p.add_argument(
        "--playoffs",
        action="store_true",
        help="Process the playoffs raw files (boxscore_raw_playoffs.parquet etc).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(playoffs=args.playoffs, verbose=True)
