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
    python src/data/features.py --playoffs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Decay rate for the exponential fatigue model (per day).
# At λ=0.25 a game played 5 days ago contributes e^(-0.25) ≈ 22% of its original load;
# Tune between 0.15 (slow decay) and 0.3 (fast decay).
FATIGUE_LAMBDA: float = 0.25

# Elo rating constants.  Starting rating and K-factor are the classic defaults;
# home-court advantage is team-specific, scaled from each team's season-to-date
# home win rate.
ELO_INITIAL: float = 1500.0
ELO_K: float = 25.0
# Home-court advantage is a linear function of the home team's home win rate:
#   home_adv = HOME_ADV_BASE + (home_win_rate − 0.5) × HOME_ADV_SCALE
# A .500 home record → 40 pts; 100% home record → 42.5 pts; 0% home record → 37.5 pts (floored by MIN=15).
ELO_HOME_ADV_BASE: float = 40.0   # advantage for a team with .500 home record
ELO_HOME_ADV_SCALE: float = 5.0  # sensitivity to home win rate
ELO_HOME_ADV_MIN: float = 15.0     # floor: worst home team still gets this bonus
# Margin-of-victory multiplier: effective K = K × log1p(|margin|)/log1p(MOV_BASELINE) × ot_factor
# MOV_BASELINE is the point margin that yields a multiplier of exactly 1.0 (≈ median NBA win margin).
ELO_MOV_BASELINE: float = 8.0
# OT discount: ot_factor = 1 / (1 + num_ot × OT_DISCOUNT)
ELO_OT_DISCOUNT: float = 0.1
# Prior-season Elo carryover: fraction of a team's deviation from 1500 that carries over.
# 0.55 → a team at 1600 starts the new season at 1555.
ELO_CARRYOVER: float = 0.55

_HERE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_HERE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERE_PROJECT_ROOT))

from src.utils.display import print_table  # noqa: E402
from src.utils.io import PROJECT_ROOT, SEASON, configure_logging, read_interim, read_parquet, write_processed  # noqa: E402

log = configure_logging("features")


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------

def _load_prior_season_elo(carryover: float = ELO_CARRYOVER) -> dict[int, float] | None:
    """Load each team's final Elo from the prior season and regress toward 1500.

    Prefers the playoffs snapshot (``elo_ratings_playoffs.parquet``) so carry-over
    reflects the true end-of-season strength, falling back to the regular-season
    snapshot when no playoff file exists.  Returns None if no prior data is found.

    With ``carryover=0.55`` a team that finished at 1600 starts the new season at
    1500 + 0.55 * (1600 - 1500) = 1555, carrying over 55% of the deviation.
    """
    prior_season = str(int(SEASON) - 1)
    prior_dir = PROJECT_ROOT / "data" / prior_season / "processed"

    for fname in ("elo_ratings_playoffs.parquet", "elo_ratings.parquet"):
        path = prior_dir / fname
        if path.exists():
            elo_ts = read_parquet(path)
            final = (
                elo_ts.sort_values("game_date")
                .groupby("team_id")["elo_post"]
                .last()
            )
            regressed = ELO_INITIAL + (carryover) * (final - ELO_INITIAL)
            log.info(
                "_load_prior_season_elo: loaded %d team ratings from %s (carryover=%.0f%%)",
                len(regressed), path.relative_to(PROJECT_ROOT), carryover * 100,
            )
            return regressed.to_dict()

    log.info("_load_prior_season_elo: no prior-season Elo found in %s — using flat initial", prior_dir)
    return None


def compute_elo_ratings(
    game_log: pd.DataFrame,
    initial: float = ELO_INITIAL,
    k: float = ELO_K,
    home_adv_base: float = ELO_HOME_ADV_BASE,
    home_adv_scale: float = ELO_HOME_ADV_SCALE,
    home_adv_min: float = ELO_HOME_ADV_MIN,
    mov_baseline: float = ELO_MOV_BASELINE,
    ot_discount: float = ELO_OT_DISCOUNT,
    initial_ratings: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Compute pre- and post-game Elo ratings for every team-game row.

    Games are replayed in chronological order.  Each team starts at
    ``initial`` and its rating is updated after every game using a
    margin-of-victory-adjusted Elo formula::

        expected_home = 1 / (1 + 10 ** ((R_away − (R_home + H)) / 400))
        mov_mult      = log1p(|margin|) / log1p(mov_baseline)
        ot_factor     = 1 / (1 + num_ot × ot_discount)
        R'            = R + K · mov_mult · ot_factor · (S − expected)

    where ``S`` is 1 for a win, 0 for a loss.  ``mov_mult`` scales the update
    with the final point margin on a log scale — a win by ``mov_baseline``
    points gives exactly 1.0 (calibrated to the typical NBA win margin of ~8
    pts), a blowout win by 30 gives ~1.56×, and a 1-point win gives ~0.32×.
    ``ot_factor`` discounts overtime wins: a 1-point win in 2 OTs carries less
    Elo weight than the same margin in regulation.

    The home-court bonus ``H`` is *team-specific*::

        H = max(home_adv_min, home_adv_base + (home_win_rate − 0.5) × home_adv_scale)

    Teams with no prior home games start at the .500 prior (``home_adv_base``).

    ``elo_pre`` is the rating carried *into* the current game (safe as a
    model feature); ``elo_post`` is the updated rating after the result.

    Parameters
    ----------
    game_log:
        Team game-log with columns ``game_id``, ``game_date``, ``team_id``,
        ``is_home``, ``win``, ``plus_minus``, ``num_ot``.  Two rows per game.
    initial, k:
        Starting Elo and base update step size.
    home_adv_base, home_adv_scale, home_adv_min:
        Parameters controlling the team-specific home-court bonus.
    mov_baseline:
        Point margin that yields a MOV multiplier of exactly 1.0.
    ot_discount:
        Fractional K reduction per overtime period.
    initial_ratings:
        Optional per-team starting ratings (keyed by ``team_id`` int).  Teams not
        present fall back to ``initial``.  Pass the output of
        ``_load_prior_season_elo()`` to carry over regressed end-of-season ratings.

    Returns
    -------
    pd.DataFrame
        One row per team per game with columns ``game_id``, ``team_id``,
        ``elo_pre``, ``elo_post``, ``home_adv``.
    """
    want_cols = ["game_id", "game_date", "team_id", "is_home", "win", "plus_minus", "num_ot"]
    cols = [c for c in want_cols if c in game_log.columns]
    games = game_log[cols].sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings: dict[int, float] = {}
    home_games: dict[int, int] = {}   # home games played before current game
    home_wins: dict[int, int] = {}    # home wins before current game
    records: list[dict] = []

    log_baseline = np.log1p(mov_baseline)

    skipped: list[str] = []
    neutral_site_count: int = 0
    # groupby with sort=False preserves the chronological order established above.
    for game_id, group in games.groupby("game_id", sort=False):
        home_row = group[group["is_home"]]
        away_row = group[~group["is_home"]]

        neutral_site = False
        if home_row.empty or away_row.empty:
            # Neutral-site game: both rows carry is_home=False (e.g. In-Season
            # Tournament finals, Mexico City games).  Process with home_adv=0
            # and do not count toward either team's home record.
            if len(group) == 2 and home_row.empty:
                home_row = group.iloc[[0]]
                away_row = group.iloc[[1]]
                neutral_site = True
                neutral_site_count += 1
            else:
                skipped.append(str(game_id))
                continue

        home = home_row.iloc[0]
        away = away_row.iloc[0]

        h_id = int(home["team_id"])
        a_id = int(away["team_id"])
        _fallback = initial_ratings or {}
        r_h = ratings.get(h_id, _fallback.get(h_id, initial))
        r_a = ratings.get(a_id, _fallback.get(a_id, initial))

        # Team-specific home advantage from prior home record (.500 prior for debut).
        # Neutral-site games get no home advantage.
        if neutral_site:
            home_adv = 0.0
        else:
            h_played = home_games.get(h_id, 0)
            home_win_rate = home_wins.get(h_id, 0) / h_played if h_played > 0 else 0.5
            home_adv = max(home_adv_min, home_adv_base + (home_win_rate - 0.5) * home_adv_scale)

        exp_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + home_adv)) / 400.0))
        s_h = 1.0 if bool(home["win"]) else 0.0

        # Margin-of-victory multiplier (log-scaled, normalised to 1.0 at mov_baseline).
        abs_margin = abs(int(home["plus_minus"])) if "plus_minus" in home.index else 0
        mov_mult = np.log1p(abs_margin) / log_baseline if log_baseline > 0 else 1.0

        # Overtime discount: each extra period reduces the effective K.
        n_ot = int(home["num_ot"]) if ("num_ot" in home.index and pd.notna(home["num_ot"])) else 0
        ot_factor = 1.0 / (1.0 + n_ot * ot_discount)

        eff_k = k * mov_mult * ot_factor

        new_r_h = r_h + eff_k * (s_h - exp_h)
        new_r_a = r_a + eff_k * ((1.0 - s_h) - (1.0 - exp_h))

        game_date = home["game_date"]
        records.append({"game_id": game_id, "game_date": game_date, "team_id": h_id,
                        "elo_pre": r_h, "elo_post": new_r_h, "home_adv": home_adv})
        records.append({"game_id": game_id, "game_date": game_date, "team_id": a_id,
                        "elo_pre": r_a, "elo_post": new_r_a, "home_adv": None})

        ratings[h_id] = new_r_h
        ratings[a_id] = new_r_a
        if not neutral_site:
            h_played = home_games.get(h_id, 0)
            home_games[h_id] = h_played + 1
            if s_h == 1.0:
                home_wins[h_id] = home_wins.get(h_id, 0) + 1

    if neutral_site_count:
        log.info(
            "compute_elo_ratings: processed %d neutral-site game(s) with home_adv=0",
            neutral_site_count,
        )
    if skipped:
        preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        log.warning(
            "compute_elo_ratings: skipped %d game(s) missing a home or away row: %s",
            len(skipped),
            preview,
        )

    return pd.DataFrame.from_records(records)

def _compute_win_streak(wins: pd.Series) -> pd.Series:
    """Current win/loss streak entering each game.

    Called via groupby("team_id")["win"].transform, so the series is already
    sorted in chronological order within each team group.  Returns a float
    Series (positive = win streak, negative = loss streak, 0 = no history).
    """
    out = np.zeros(len(wins))
    streak = 0
    for i, v in enumerate(wins):
        out[i] = streak
        if pd.isna(v):
            streak = 0
        elif int(v) == 1:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
    return pd.Series(out, index=wins.index)


def build_team_features(
    game_log: pd.DataFrame,
    team_advanced: pd.DataFrame,
    initial_elo_ratings: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Build team-level feature table with rolling season averages.

    Merges ``game_log`` with efficiency metrics from ``team_advanced``, attaches
    opponent identity via self-join, then computes expanding-window season
    averages (shift-by-1 so the current game is excluded) plus recent-form
    windows, win/loss streak, and head-to-head win rate.

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
        ``season_win_rate``, ``games_played``, ``recent_form_*``,
        ``win_streak``, ``h2h_win_rate``, and Elo features.
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

    # --- Opponent identity and per-game opponent stats ----------------------
    # Self-join game_log on game_id: each game has 2 rows, so every team row
    # gets one opponent candidate; we keep only the non-self match.
    opp = game_log[["game_id", "team_id", "dreb", "pts", "oreb", "blk", "stl"]].rename(
        columns={
            "team_id": "opp_team_id",
            "dreb": "opp_dreb",
            "pts": "opp_pts",
            "oreb": "opp_oreb",
            "blk": "opp_blk",
            "stl": "opp_stl",
        }
    )
    df = df.merge(opp, on="game_id", how="left")
    df = df[df["team_id"] != df["opp_team_id"]].copy()

    # Real OREB%: own offensive rebounds / (own OREB + opponent DREB)
    df["true_oreb_pct"] = df["oreb"] / (df["oreb"] + df["opp_dreb"]).replace(0, np.nan)

    # Opponent OREB%: opponent's offensive rebounds / (opp OREB + own DREB)
    df["opp_oreb_pct"] = df["opp_oreb"] / (df["opp_oreb"] + df["dreb"]).replace(0, np.nan)

    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    # --- Rolling season averages (shift-1, prior games only) ---------------
    avg_cols = [
        "pts", "reb", "oreb", "ast", "stl", "blk", "tov", "pf", "fta",
        "fg_pct", "fg3_pct", "ft_pct", "plus_minus",
        "true_shooting_pct", "three_point_rate", "free_throw_rate", "oreb_pct_proxy",
        "true_oreb_pct",
        "opp_pts", "opp_oreb", "opp_oreb_pct", "opp_blk", "opp_stl",
    ]
    for col in avg_cols:
        if col in df.columns:
            grouped = df.groupby("team_id")[col]
            df[f"season_avg_{col}"] = grouped.transform(lambda s: s.shift(1).expanding().mean())
            df[f"last10_avg_{col}"] = grouped.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

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

    # --- Recent form: rolling win rate over last N games (shift-1) ---------
    for n in [5, 10, 15]:
        df[f"recent_form_{n}"] = (
            df.groupby("team_id")["win"]
            .transform(lambda s, _n=n: s.shift(1).rolling(_n, min_periods=1).mean())
        )

    # --- Win/loss streak entering each game --------------------------------
    # df is sorted by [team_id, game_date] so each group is chronological.
    df["win_streak"] = (
        df.groupby("team_id")["win"]
        .transform(_compute_win_streak)
    )

    # --- Head-to-head win rate vs this specific opponent (shift-1) ---------
    h2h_sorted = df.sort_values(["team_id", "opp_team_id", "game_date"])
    h2h_rates = (
        h2h_sorted.groupby(["team_id", "opp_team_id"])["win"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    # No prior h2h history → 0.5 (assume parity; win_rate_delta already captures
    # overall quality, so h2h_delta=0 correctly signals no matchup-specific edge).
    df["h2h_win_rate"] = h2h_rates.fillna(0.5)

    # --- Elo ratings --------------------------------------------------------
    elo = compute_elo_ratings(game_log, initial_ratings=initial_elo_ratings)
    elo_merge = elo.drop(columns=["game_date"], errors="ignore")
    df = df.merge(elo_merge, on=["game_id", "team_id"], how="left")

    # Attach opponent pre-game Elo using the already-joined opp_team_id.
    opp_elo = elo_merge[["game_id", "team_id", "elo_pre"]].rename(
        columns={"team_id": "opp_team_id", "elo_pre": "opp_elo_pre"}
    )
    df = df.merge(opp_elo, on=["game_id", "opp_team_id"], how="left")

    df = df.drop(columns=["opp_team_id", "opp_dreb", "opp_pts", "opp_oreb", "opp_blk", "opp_stl"], errors="ignore")

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

    Implementation: O(n) via cumulative products. Let D_k = Π_{i<k} e^{-λ Δ_i}
    where Δ_i = t_{i+1} - t_i. Then fatigue_i = D_i · Σ_{j<i} m_j / D_j.
    Both the cumulative product and cumulative sum are vectorised, so no
    Python-level per-game loop is required.
    """
    group = group.sort_values("game_date")
    dates = group["game_date"].values                        # datetime64[ns]
    minutes = group["minutes_decimal"].fillna(0).to_numpy(dtype=float)
    n = len(group)
    result = np.zeros(n)
    if n < 2:
        return pd.Series(result, index=group.index)

    deltas = np.diff(dates) / np.timedelta64(1, "D")         # length n-1
    decay = np.exp(-lam * deltas)                             # length n-1
    # D[k] = Π_{i=0..k-1} decay[i], with D[0] = 1.
    D = np.concatenate(([1.0], np.cumprod(decay)))            # length n
    # fatigue[i] = D[i] * cumsum_{j<i}(minutes[j] / D[j])
    m_over_D = minutes[:-1] / D[:-1]                          # length n-1
    csum = np.cumsum(m_over_D)                                # length n-1
    result[1:] = D[1:] * csum
    return pd.Series(result, index=group.index)


def _acwr_player(group: pd.DataFrame) -> pd.Series:
    """Acute:Chronic Workload Ratio for one player's chronological game sequence.

    ACWR = (7-day rolling minutes) / (28-day rolling minutes / 4)

    Both windows are *left-closed* (current game excluded), so the ratio
    reflects the load leading into the current game rather than including it.
    Returns NaN when the chronic window is empty (no prior games in 28 days).

    Same-date duplicates (rare in NBA, but possible) are preserved: the
    rolling operation is performed on a positional-index ``DataFrame`` with
    ``game_date`` as an auxiliary column, and results are aligned back to the
    original row index rather than keyed on date.
    """
    g = group[["game_date", "minutes_decimal"]].sort_values("game_date").copy()
    g["minutes_decimal"] = g["minutes_decimal"].fillna(0)
    # Rolling with a time offset requires a DatetimeIndex.  Duplicate dates
    # are allowed; pandas handles them positionally within a window.
    tmp = g.set_index(pd.DatetimeIndex(g["game_date"]))
    acute = tmp["minutes_decimal"].rolling("7D", closed="left").sum()
    chronic_weekly = tmp["minutes_decimal"].rolling("28D", closed="left").sum() / 4
    ratio = (acute / chronic_weekly.replace(0, np.nan)).to_numpy()
    # Align back to the original (un-sorted) group index via g's preserved order.
    out = pd.Series(ratio, index=g.index)
    return out.reindex(group.index)


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
    initial_elo_ratings: dict[int, float] | None = None,
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

    team_features = build_team_features(game_log, team_advanced, initial_elo_ratings=initial_elo_ratings)
    player_features = build_player_features(player_game_log, game_log)
    team_player_feats = build_team_player_features(player_features)
    team_features = team_features.merge(team_player_feats, on=["game_id", "team_id"], how="left")

    return {"team_features": team_features, "player_features": player_features}


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    cutoff_date=None,
    playoffs: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute the feature-engineering pipeline and optionally write processed tables.

    Parameters
    ----------
    cutoff_date:
        If provided, only games on or before this date are processed.  When a
        cutoff is given, nothing is written to disk (in-memory only).  Call
        without a cutoff at the end of the season to save the final snapshot
        and the Elo time series.
    playoffs:
        If True, load playoff interim files (``*_playoffs.parquet``) and
        concatenate them with the regular-season tables before computing
        features.  Regular-season data is always included so that Elo ratings
        carry over and rolling averages are not cold-started at game 1 of the
        playoffs.  Only playoff game rows are kept in the outputs and written
        to disk (``*_playoffs.parquet`` processed files).
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

    playoff_game_ids: set | None = None

    if playoffs:
        game_log_po = read_interim("game_log_playoffs.parquet")
        player_game_log_po = read_interim("player_game_log_playoffs.parquet")
        team_advanced_po = read_interim("team_advanced_playoffs.parquet")

        playoff_game_ids = set(game_log_po["game_id"])

        # Combine: regular-season rows provide rolling history; playoff rows
        # are the target.  Game IDs are disjoint so no dedup is needed.
        game_log = pd.concat([game_log, game_log_po], ignore_index=True)
        player_game_log = pd.concat([player_game_log, player_game_log_po], ignore_index=True)
        team_advanced = pd.concat([team_advanced, team_advanced_po], ignore_index=True)

    if verbose:
        label = "playoffs (reg + playoff combined)" if playoffs else "regular season"
        print(f"  game_log         : {game_log.shape[0]} rows, {game_log.shape[1]} cols  [{label}]")
        print(f"  player_game_log  : {player_game_log.shape[0]} rows, {player_game_log.shape[1]} cols")
        print(f"  team_advanced    : {team_advanced.shape[0]} rows, {team_advanced.shape[1]} cols")

    prior_elo = _load_prior_season_elo()
    if verbose and prior_elo:
        print(f"  prior-season Elo : loaded {len(prior_elo)} team ratings ({(1 - ELO_CARRYOVER) * 100:.0f}% regression toward 1500)")

    outputs = compute_features_from_data(
        game_log, player_game_log, team_advanced, cutoff_date,
        initial_elo_ratings=prior_elo,
    )

    if playoffs and playoff_game_ids is not None:
        outputs["team_features"] = (
            outputs["team_features"][outputs["team_features"]["game_id"].isin(playoff_game_ids)]
            .reset_index(drop=True)
        )
        outputs["player_features"] = (
            outputs["player_features"][outputs["player_features"]["game_id"].isin(playoff_game_ids)]
            .reset_index(drop=True)
        )

    if cutoff_date is None:
        suffix = "_playoffs" if playoffs else ""
        if verbose:
            print("\nSaving to data/processed/ ...")

        for name, df in outputs.items():
            dest = write_processed(df, f"{name}{suffix}.parquet")
            if verbose:
                print(f"  -> {dest.relative_to(PROJECT_ROOT)}")
            log.info("%s%s: %d rows written", name, suffix, len(df))

        # Elo time series: one row per team per game.
        elo_ts = outputs["team_features"][
            ["game_id", "game_date", "team_id", "elo_pre", "elo_post"]
        ].copy()
        dest = write_processed(elo_ts, f"elo_ratings{suffix}.parquet")
        if verbose:
            print(f"  -> {dest.relative_to(PROJECT_ROOT)}")
        log.info("elo_ratings%s: %d rows written", suffix, len(elo_ts))

        tf = outputs["team_features"]
        for src_col, delta_name in [
            ("elo_pre", "elo_delta"),
            ("team_fatigue", "fatigue_delta"),
            ("h2h_win_rate", "h2h_delta"),
        ]:
            if src_col in tf.columns:
                nan_rate = tf[src_col].isna().mean() * 100
                log.info("NaN rate for %s (-> %s): %.2f%%", src_col, delta_name, nan_rate)

    if verbose and cutoff_date is None:
        print_table("team_features", outputs["team_features"])
        print_table("player_features", outputs["player_features"])

    return outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Feature engineering pipeline for NBA stats model.",
    )
    p.add_argument(
        "--cutoff",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Only use games on or before this date.  Features for games on the "
            "cutoff date are computed from strictly prior games (shift-1), so "
            "the result is the correct entering-game state.  When omitted the "
            "full season is processed and results are written to disk."
        ),
    )
    p.add_argument(
        "--playoffs",
        action="store_true",
        help=(
            "Process playoff interim files (*_playoffs.parquet).  Regular-season "
            "data is always included as rolling history so Elo and averages carry "
            "over.  Outputs are filtered to playoff games and written to "
            "*_playoffs.parquet processed files."
        ),
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(cutoff_date=args.cutoff, playoffs=args.playoffs, verbose=True)
