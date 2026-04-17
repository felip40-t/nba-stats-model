"""
features.py — Stage 3 of the NBA stats pipeline.

Reads interim tables from ``data/interim/``, engineers features, and writes
processed tables to ``data/processed/``.

Feature sets
------------
team_features.parquet
    One row per team per game.  Contains all game_log columns plus advanced
    metrics from team_advanced, plus expanding-window (season-to-date) rolling
    averages for key stats computed from games *prior* to the current one.

player_features.parquet
    One row per player per game.  Contains all player_game_log columns plus
    season-to-date rolling averages and a fatigue metric.

Fatigue metrics
---------------
Two complementary player load/fatigue metrics are computed from prior games
only (current game is never included):

``fatigue_decay`` — exponential decay model
    fatigue_i = Σ_{j<i}  minutes_j · e^{−λ·(date_i − date_j)}
    where λ = FATIGUE_LAMBDA (default 0.2 day⁻¹).  Recent high-minute games
    contribute more; load from distant games fades exponentially with rest.

``acwr`` — Acute:Chronic Workload Ratio
    acwr_i = (7-day rolling minutes) / (28-day rolling minutes / 4)
    Values > 1 signal a spike in acute load relative to chronic baseline,
    which is associated with elevated injury risk in sports science literature.

Run from the project root::

    python src/data/features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Decay rate for the exponential fatigue model (per day).
# At λ=0.2 a game played 5 days ago contributes e^(-1) ≈ 37% of its original load;
# 10 days ago ≈ 14%.  Tune between 0.15 (slow decay) and 0.3 (fast decay).
FATIGUE_LAMBDA: float = 0.2

# Elo rating constants.  Starting rating and K-factor are the classic defaults;
# home-court advantage is team-specific, scaled from each team's season-to-date
# home win rate.  All are placeholders — calibrate once we have full-season data.
ELO_INITIAL: float = 1500.0
ELO_K: float = 20.0
# Home-court advantage is a linear function of the home team's home win rate:
#   home_adv = HOME_ADV_BASE + (home_win_rate − 0.5) × HOME_ADV_SCALE
# A .500 home record → 100 pts; 100% → 150 pts; 0% → 50 pts (floored by MIN).
ELO_HOME_ADV_BASE: float = 100.0   # advantage for a team with .500 home record
ELO_HOME_ADV_SCALE: float = 100.0  # sensitivity to home win rate
ELO_HOME_ADV_MIN: float = 50.0     # floor: worst home team still gets this bonus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_interim, write_processed  # noqa: E402


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------

def compute_elo_ratings(
    game_log: pd.DataFrame,
    initial: float = ELO_INITIAL,
    k: float = ELO_K,
    home_adv_base: float = ELO_HOME_ADV_BASE,
    home_adv_scale: float = ELO_HOME_ADV_SCALE,
    home_adv_min: float = ELO_HOME_ADV_MIN,
) -> pd.DataFrame:
    """Compute pre- and post-game Elo ratings for every team-game row.

    Games are replayed in chronological order.  Each team starts at
    ``initial`` and its rating is updated after every game using the
    classic Elo formula::

        expected_home = 1 / (1 + 10 ** ((R_away − (R_home + H)) / 400))
        R'            = R + K · (S − expected)

    where ``S`` is 1 for a win, 0 for a loss.  The home-court bonus ``H``
    is *team-specific*: it scales with the home team's season-to-date home
    win rate (from prior home games only — no leakage)::

        H = max(home_adv_min, home_adv_base + (home_win_rate − 0.5) × home_adv_scale)

    A team with a .500 home record gets the baseline advantage; better home
    teams earn a higher bonus, worse home teams are floored at ``home_adv_min``.
    Teams with no prior home games start at the .500 prior (``home_adv_base``).

    ``elo_pre`` is the rating carried *into* the current game (safe as a
    model feature); ``elo_post`` is the updated rating after the result.

    Parameters
    ----------
    game_log:
        Team game-log with columns ``game_id``, ``game_date``, ``team_id``,
        ``is_home``, ``win``.  Expected to contain two rows per game.
    initial, k:
        Starting Elo and update step size.
    home_adv_base, home_adv_scale, home_adv_min:
        Parameters controlling the team-specific home-court bonus.

    Returns
    -------
    pd.DataFrame
        One row per team per game with columns ``game_id``, ``team_id``,
        ``elo_pre``, ``elo_post``, ``home_adv``.
    """
    cols = ["game_id", "game_date", "team_id", "is_home", "win"]
    games = game_log[cols].sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings: dict[int, float] = {}
    home_games: dict[int, int] = {}   # home games played before current game
    home_wins: dict[int, int] = {}    # home wins before current game
    records: list[dict] = []

    # groupby with sort=False preserves the chronological order established above.
    for game_id, group in games.groupby("game_id", sort=False):
        home_row = group[group["is_home"]]
        away_row = group[~group["is_home"]]
        if home_row.empty or away_row.empty:
            continue
        home = home_row.iloc[0]
        away = away_row.iloc[0]

        h_id = int(home["team_id"])
        a_id = int(away["team_id"])
        r_h = ratings.get(h_id, initial)
        r_a = ratings.get(a_id, initial)

        # Team-specific home advantage from prior home record (.500 prior for debut).
        h_played = home_games.get(h_id, 0)
        home_win_rate = home_wins.get(h_id, 0) / h_played if h_played > 0 else 0.5
        home_adv = max(home_adv_min, home_adv_base + (home_win_rate - 0.5) * home_adv_scale)

        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + home_adv)) / 400.0))
        s_h = 1.0 if bool(home["win"]) else 0.0

        new_r_h = r_h + k * (s_h - exp_h)
        new_r_a = r_a + k * ((1.0 - s_h) - (1.0 - exp_h))

        game_date = home["game_date"]
        records.append({"game_id": game_id, "game_date": game_date, "team_id": h_id,
                        "elo_pre": r_h, "elo_post": new_r_h, "home_adv": home_adv})
        records.append({"game_id": game_id, "game_date": game_date, "team_id": a_id,
                        "elo_pre": r_a, "elo_post": new_r_a, "home_adv": None})

        ratings[h_id] = new_r_h
        ratings[a_id] = new_r_a
        home_games[h_id] = h_played + 1
        if s_h == 1.0:
            home_wins[h_id] = home_wins.get(h_id, 0) + 1

    return pd.DataFrame.from_records(records)

def build_team_features(
    game_log: pd.DataFrame,
    team_advanced: pd.DataFrame,
) -> pd.DataFrame:
    """Build team-level feature table with rolling season averages.

    Merges ``game_log`` with the efficiency metrics from ``team_advanced``,
    then computes expanding-window season averages (shift-by-1 so the current
    game is excluded) for a set of box-score and efficiency columns.

    Parameters
    ----------
    game_log:
        Cleaned team game-log from ``data/interim/game_log.parquet``.
    team_advanced:
        Per-game team totals/efficiency from
        ``data/interim/team_advanced.parquet``.

    Returns
    -------
    pd.DataFrame
        One row per team per game with original columns plus ``season_avg_*``,
        ``season_win_rate``, and ``games_played`` features.
    """
    adv_cols = [
        "game_id", "team_id",
        "true_shooting_pct", "three_point_rate", "free_throw_rate", "oreb_pct_proxy",
    ]
    df = game_log.merge(
        team_advanced[adv_cols],
        on=["game_id", "team_id"],
        how="left",
    )

    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # Columns to roll — box-score stats + efficiency metrics
    avg_cols = [
        "pts", "reb", "ast", "stl", "blk", "tov",
        "fg_pct", "fg3_pct", "ft_pct", "plus_minus",
        "true_shooting_pct", "three_point_rate", "free_throw_rate", "oreb_pct_proxy",
    ]

    for col in avg_cols:
        if col in df.columns:
            df[f"season_avg_{col}"] = (
                df.groupby("team_id")[col]
                .transform(lambda s: s.shift(1).expanding().mean())
            )

    # Season win rate before current game
    df["season_win_rate"] = (
        df.groupby("team_id")["win"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    # Games played entering this game (first game → 0)
    df["games_played"] = (
        df.groupby("team_id")["game_id"]
        .transform(lambda s: s.shift(1).expanding().count())
    ).fillna(0).astype(int)

    # --- Elo ratings --------------------------------------------------------
    # Merge the team's own pre/post Elo, then self-join on game_id to attach
    # the opponent's pre-game Elo (useful for win-probability modelling).
    elo = compute_elo_ratings(game_log)
    df = df.merge(elo, on=["game_id", "team_id"], how="left")

    opp_elo = elo.rename(
        columns={"team_id": "opp_team_id", "elo_pre": "opp_elo_pre"}
    )[["game_id", "opp_team_id", "opp_elo_pre"]]
    df = df.merge(opp_elo, on="game_id", how="left")
    df = df[df["team_id"] != df["opp_team_id"]].drop(columns=["opp_team_id"])

    return df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Player features
# ---------------------------------------------------------------------------

def _days_since_last(dates: pd.Series) -> pd.Series:
    """Return calendar days since the previous entry (NaN for first game)."""
    return dates.diff().dt.days


def _fatigue_decay_player(group: pd.DataFrame, lam: float) -> pd.Series:
    """Exponential decay fatigue for one player's chronological game sequence.

    For game i:  fatigue_i = Σ_{j < i}  minutes_j · e^{-λ · (date_i − date_j)}

    The current game's minutes are *not* included (look-back only).
    DNP rows (NaN minutes) contribute 0 load.

    Parameters
    ----------
    group:
        Single-player slice of the player features DataFrame, sorted by date.
    lam:
        Decay rate constant (days⁻¹).
    """
    group = group.sort_values("game_date")
    dates = group["game_date"].values               # numpy datetime64[ns]
    minutes = group["minutes_decimal"].fillna(0).values
    n = len(group)
    result = np.zeros(n)
    for i in range(1, n):
        days = (dates[i] - dates[:i]) / np.timedelta64(1, "D")
        result[i] = float(np.dot(minutes[:i], np.exp(-lam * days)))
    return pd.Series(result, index=group.index)


def _acwr_player(group: pd.DataFrame) -> pd.Series:
    """Acute:Chronic Workload Ratio for one player's chronological game sequence.

    ACWR = (7-day rolling minutes) / (28-day rolling minutes / 4)

    Both windows are *left-closed* (current game excluded), so the ratio
    reflects the load leading into the current game rather than including it.
    Returns NaN when the chronic window is empty (no prior games in 28 days).

    Parameters
    ----------
    group:
        Single-player slice sorted by date, with a ``game_date`` column and
        a ``minutes_decimal`` column.
    """
    g = group.set_index("game_date")["minutes_decimal"].fillna(0).sort_index()
    acute = g.rolling("7D", closed="left").sum()
    chronic_weekly = g.rolling("28D", closed="left").sum() / 4
    ratio = acute / chronic_weekly.replace(0, np.nan)
    # ratio is indexed by game_date; map back to the original positional index
    date_to_ratio = ratio.to_dict()
    return group["game_date"].map(date_to_ratio)


def build_player_features(
    player_game_log: pd.DataFrame,
    game_log: pd.DataFrame,
) -> pd.DataFrame:
    """Build player-level feature table with rolling averages and fatigue.

    Parameters
    ----------
    player_game_log:
        Cleaned player game-log from
        ``data/interim/player_game_log.parquet``.
    game_log:
        Used only to supply ``game_date`` for each (game_id, team_id) pair.

    Returns
    -------
    pd.DataFrame
        One row per player per game with original columns plus ``season_avg_*``,
        ``days_rest``, ``fatigue_decay``, and ``acwr``.
    """
    # Attach game_date from game_log (one row per game_id/team_id pairing)
    date_map = (
        game_log[["game_id", "team_id", "game_date"]]
        .drop_duplicates(subset=["game_id", "team_id"])
    )
    df = player_game_log.merge(date_map, on=["game_id", "team_id"], how="left")

    df = df.sort_values(["person_id", "game_date"]).reset_index(drop=True)

    # --- Rolling season averages (prior games only) -------------------------
    avg_cols = [
        "points", "rebounds_total", "assists", "steals", "blocks",
        "turnovers", "minutes_decimal",
        "field_goals_percentage", "three_pointers_percentage",
        "free_throws_percentage", "plus_minus_points",
    ]

    for col in avg_cols:
        if col in df.columns:
            df[f"season_avg_{col}"] = (
                df.groupby("person_id")[col]
                .transform(lambda s: s.shift(1).expanding().mean())
            )

    # --- Fatigue metrics ----------------------------------------------------
    # Days since last game (NaN for season debut)
    df["days_rest"] = (
        df.groupby("person_id")["game_date"]
        .transform(_days_since_last)
    )

    # 1. Exponential decay load: Σ mj · e^{-λ(ti − tj)} over all prior games j
    df["fatigue_decay"] = (
        df.groupby("person_id", group_keys=False)
        .apply(lambda g: _fatigue_decay_player(g, FATIGUE_LAMBDA))
    )

    # 2. Acute:Chronic Workload Ratio — 7-day minutes / (28-day minutes / 4)
    df["acwr"] = (
        df.groupby("person_id", group_keys=False)
        .apply(_acwr_player)
    )

    # --- Tidy column order --------------------------------------------------
    front_cols = [
        "game_id", "game_date",
        "team_id", "team_tricode", "team_city", "team_name",
        "person_id", "first_name", "family_name", "name_i",
        "position", "jersey_num",
        "minutes", "minutes_decimal",
        "days_rest", "fatigue_decay", "acwr",
    ]
    remaining = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining]

    return df.sort_values(["game_date", "game_id", "person_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Team-level player aggregates
# ---------------------------------------------------------------------------

def build_team_player_features(player_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player fatigue metrics to the team-game level.

    Each metric is a minutes-weighted average across players.  The weight for
    player *i* is ``season_avg_minutes_decimal`` — their season average minutes
    entering this game (shift-by-1, so DNP / debut players have NaN weight and
    are excluded from the average).

    Returns
    -------
    pd.DataFrame
        One row per (game_id, team_id) with columns ``team_fatigue`` and
        ``team_acwr``.  NaN when no player on the team has a valid weight for
        that game (typically the first game of the season for each team).
    """
    df = player_features[
        ["game_id", "team_id", "season_avg_minutes_decimal", "fatigue_decay", "acwr"]
    ].copy()

    # Keep only players with a positive average-minutes weight
    df = df[df["season_avg_minutes_decimal"].notna() & (df["season_avg_minutes_decimal"] > 0)].copy()

    # For each metric, compute w*v and w separately (setting NaN where value is NaN
    # so that pandas .sum(skipna=True) naturally excludes those players)
    for col, out in [("fatigue_decay", "team_fatigue"), ("acwr", "team_acwr")]:
        mask = df[col].notna()
        df[f"_wv_{col}"] = np.where(mask, df["season_avg_minutes_decimal"] * df[col], np.nan)
        df[f"_wt_{col}"] = np.where(mask, df["season_avg_minutes_decimal"], np.nan)

    agg = df.groupby(["game_id", "team_id"]).agg(
        _wv_fatigue=("_wv_fatigue_decay", "sum"),
        _wt_fatigue=("_wt_fatigue_decay", "sum"),
        _wv_acwr=("_wv_acwr", "sum"),
        _wt_acwr=("_wt_acwr", "sum"),
    ).reset_index()

    agg["team_fatigue"] = agg["_wv_fatigue"] / agg["_wt_fatigue"].replace(0, np.nan)
    agg["team_acwr"] = agg["_wv_acwr"] / agg["_wt_acwr"].replace(0, np.nan)

    return agg[["game_id", "team_id", "team_fatigue", "team_acwr"]]


# ---------------------------------------------------------------------------
# In-memory feature computation (no disk I/O)
# ---------------------------------------------------------------------------

def compute_features_from_data(
    game_log: pd.DataFrame,
    player_game_log: pd.DataFrame,
    team_advanced: pd.DataFrame,
    cutoff_date=None,
) -> dict[str, pd.DataFrame]:
    """Compute team and player features from pre-loaded interim tables.

    No disk I/O — callers supply data directly.  This is the workhorse used
    by both ``run_pipeline`` and the rolling training simulation in
    ``train.py``.

    Parameters
    ----------
    game_log, player_game_log, team_advanced:
        Interim tables as loaded by ``read_interim``.
    cutoff_date:
        If provided, only games on or before this date are used.  Because
        rolling stats use ``shift(1)``, a game row on the cutoff date carries
        features derived solely from games strictly before that date — i.e.
        the correct "entering-game" state for prediction on that date.

    Returns
    -------
    dict with keys ``"team_features"`` and ``"player_features"``.
    """
    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
        game_log = game_log[game_log["game_date"] <= cutoff].copy()
        valid_ids = set(game_log["game_id"])
        player_game_log = player_game_log[player_game_log["game_id"].isin(valid_ids)].copy()

    team_features = build_team_features(game_log, team_advanced)
    player_features = build_player_features(player_game_log, game_log)
    team_player_feats = build_team_player_features(player_features)
    team_features = team_features.merge(team_player_feats, on=["game_id", "team_id"], how="left")

    return {"team_features": team_features, "player_features": player_features}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table(title: str, df: pd.DataFrame, max_cols: int = 12) -> None:
    """Pretty-print a DataFrame with a heading, splitting wide tables into
    column blocks so they fit a standard terminal."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(sep)
    cols = list(df.columns)
    for start in range(0, len(cols), max_cols):
        chunk = cols[start: start + max_cols]
        print(df[chunk].to_string(index=True))
        if start + max_cols < len(cols):
            print()


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(cutoff_date=None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Execute the feature-engineering pipeline and optionally write processed tables.

    Parameters
    ----------
    cutoff_date:
        If provided, only games on or before this date are processed.  When a
        cutoff is given, nothing is written to disk (in-memory only).  Call
        without a cutoff at the end of the season to save the final snapshot
        and the Elo time series.
    verbose:
        If True, print progress, table shapes, and previews.

    Returns
    -------
    dict with keys ``"team_features"`` and ``"player_features"``.
    """
    if verbose:
        print("Loading interim data...")

    game_log = read_interim("game_log.parquet")
    player_game_log = read_interim("player_game_log.parquet")
    team_advanced = read_interim("team_advanced.parquet")

    if verbose:
        print(f"  game_log         : {game_log.shape[0]} rows, {game_log.shape[1]} cols")
        print(f"  player_game_log  : {player_game_log.shape[0]} rows, {player_game_log.shape[1]} cols")
        print(f"  team_advanced    : {team_advanced.shape[0]} rows, {team_advanced.shape[1]} cols")

    outputs = compute_features_from_data(game_log, player_game_log, team_advanced, cutoff_date)

    if cutoff_date is None:
        if verbose:
            print("\nSaving to data/processed/ ...")

        for name, df in outputs.items():
            dest = write_processed(df, f"{name}.parquet")
            if verbose:
                print(f"  -> {dest.relative_to(PROJECT_ROOT)}")

        # Elo time series: one row per team per game for the full season.
        elo_ts = outputs["team_features"][
            ["game_id", "game_date", "team_id", "elo_pre", "elo_post"]
        ].copy()
        dest = write_processed(elo_ts, "elo_ratings.parquet")
        if verbose:
            print(f"  -> {dest.relative_to(PROJECT_ROOT)}")

    if verbose and cutoff_date is None:
        _print_table("team_features", outputs["team_features"])
        _print_table("player_features", outputs["player_features"])

    return outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(verbose=True)
