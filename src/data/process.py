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
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the project root is on sys.path so ``src`` is importable when
# the script is run directly (python src/data/process.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_raw, write_interim  # noqa: E402


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


# ---------------------------------------------------------------------------
# game_log
# ---------------------------------------------------------------------------

def clean_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and restructure the raw team game-log DataFrame.

    Standardises column names, parses dates, casts dtypes, and derives
    three contextual flags: ``is_home``, ``win``, and ``is_back_to_back``.

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

    int_cols = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                "oreb", "dreb", "reb", "ast", "stl", "blk",
                "tov", "pf", "pts", "plus_minus", "min"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = ["fg_pct", "fg3_pct", "ft_pct"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
    With only a single game in the dataset every team's flag will be False;
    the logic is correct for multi-game datasets.
    """
    df = df.copy()
    df = df.sort_values(["team_id", "game_date"])
    prev_date = df.groupby("team_id")["game_date"].shift(1)
    df["is_back_to_back"] = (df["game_date"] - prev_date).dt.days == 1
    return df


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

    int_cols = ["field_goals_made", "field_goals_attempted",
                "three_pointers_made", "three_pointers_attempted",
                "free_throws_made", "free_throws_attempted",
                "rebounds_offensive", "rebounds_defensive", "rebounds_total",
                "assists", "steals", "blocks", "turnovers", "fouls_personal", "points"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = ["field_goals_percentage", "three_pointers_percentage",
                  "free_throws_percentage", "plus_minus_points"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
    for col in sum_cols:
        if col in team_df.columns:
            team_df[col] = pd.to_numeric(team_df[col], errors="coerce")

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
    for col in sum_cols:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].astype("Int64")

    return agg_df.sort_values(["game_id", "team_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table(title: str, df: pd.DataFrame, max_cols: int = 12) -> None:
    """Pretty-print a DataFrame to stdout with a heading.

    Shows all rows but limits visible columns to ``max_cols`` per block so
    the output stays readable in a terminal without horizontal scrolling.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(sep)

    cols = list(df.columns)
    for start in range(0, len(cols), max_cols):
        chunk = cols[start: start + max_cols]
        print(df[chunk].to_string(index=True))
        if start + max_cols < len(cols):
            print()  # blank line between column blocks


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Execute the full processing pipeline and return the three output tables.

    Reads raw files, applies all cleaning and transformation functions, saves
    the results to ``data/interim/``, and (when ``verbose=True``) prints a
    formatted preview of every output table.

    Parameters
    ----------
    verbose:
        If True, print table previews and file paths to stdout.

    Returns
    -------
    dict with keys ``"game_log"``, ``"player_game_log"``, ``"team_advanced"``.
    """
    if verbose:
        print("Loading raw data...")

    gamelog_raw = read_raw("team_gamelog_raw.parquet")
    boxscore_raw = read_raw("boxscore_raw.parquet")

    if verbose:
        print(f"  team_gamelog_raw : {gamelog_raw.shape[0]} rows, "
              f"{gamelog_raw.shape[1]} cols")
        print(f"  boxscore_raw     : {boxscore_raw.shape[0]} rows, "
              f"{boxscore_raw.shape[1]} cols")

    # --- transform ----------------------------------------------------------
    game_log = clean_game_log(gamelog_raw)
    player_game_log = clean_player_game_log(boxscore_raw)
    team_advanced = build_team_advanced(boxscore_raw)

    # --- save ---------------------------------------------------------------
    outputs = {
        "game_log": game_log,
        "player_game_log": player_game_log,
        "team_advanced": team_advanced,
    }

    if verbose:
        print("\nSaving to data/interim/ ...")

    for name, df in outputs.items():
        dest = write_interim(df, f"{name}.parquet")
        if verbose:
            print(f"  -> {dest.relative_to(PROJECT_ROOT)}")

    # --- display ------------------------------------------------------------
    if verbose:
        _print_table("game_log", game_log)
        _print_table("player_game_log", player_game_log)
        _print_table("team_advanced", team_advanced)

    return outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(verbose=True)
