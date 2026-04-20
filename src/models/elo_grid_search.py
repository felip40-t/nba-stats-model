"""
elo_grid_search.py  —  Grid search over Elo hyperparameters to minimise log-loss.

Run from the project root::

    python src/models/elo_grid_search.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.display import print_table
from src.utils.io import OUTPUTS_DIR, PROJECT_ROOT, SEASON, read_interim, read_parquet, write_parquet

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ELO_INITIAL: float = 1500.0
COLD_START_GAMES: int = 50  # first N games skipped from log-loss evaluation

GRID: dict[str, list[float]] = {
    "carryover":      [0.45, 0.5, 0.55],
    "k":              [20.0, 25.0, 30.0],
    "home_adv_base":  [30.0, 40.0, 50.0],
    "home_adv_scale": [5.0, 10.0, 15.0, 20.0,],
    "home_adv_min":   [15.0, 20.0, 25.0],
    "mov_baseline":   [7.0, 8.0, 9.0],
    "ot_discount":    [0.1, 0.2, 0.3,],
    "inv_scale":      [100.0, 200.0, 300.0, 400.0, 500.0, 600.0], 
}



# ---------------------------------------------------------------------------
# Prior season helpers
# ---------------------------------------------------------------------------

def _load_prior_final_elo() -> dict[int, float] | None:
    """Return each team's final elo_post from the prior season, or None."""
    prior_dir = PROJECT_ROOT / "data" / str(int(SEASON) - 1) / "processed"
    for fname in ("elo_ratings_playoffs.parquet", "elo_ratings.parquet"):
        path = prior_dir / fname
        if path.exists():
            ts = read_parquet(path)
            final = ts.sort_values("game_date").groupby("team_id")["elo_post"].last()
            return final.to_dict()
    return None


def _make_initial_ratings(
    prior_final: dict[int, float] | None,
    carryover: float,
) -> dict[int, float] | None:
    """Apply season-to-season regression toward ELO_INITIAL."""
    if carryover == 0.0 or prior_final is None:
        return None
    return {
        tid: ELO_INITIAL + carryover * (elo - ELO_INITIAL)
        for tid, elo in prior_final.items()
    }


# ---------------------------------------------------------------------------
# Elo simulation (self-contained — does not import from features.py)
# ---------------------------------------------------------------------------

def compute_elo(
    game_log: pd.DataFrame,
    params: dict,
    initial_ratings: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Simulate Elo ratings chronologically with the given hyperparameters.

    Parameters
    ----------
    game_log:
        Interim table with columns game_id, game_date, team_id, is_home,
        win, plus_minus, num_ot.
    params:
        Keys: k, home_adv_base, home_adv_scale, home_adv_min, mov_baseline,
        ot_discount, inv_scale.  ``carryover`` is applied externally via initial_ratings.
    initial_ratings:
        Per-team starting Elo (already regressed for the desired carryover).
        Teams absent here start at ELO_INITIAL.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, team_id, game_date, elo_pre, home_adv.
        home_adv is NaN for away-team rows.
    """
    k           = float(params["k"])
    adv_base    = float(params["home_adv_base"])
    adv_scale   = float(params["home_adv_scale"])
    adv_min     = float(params["home_adv_min"])
    log_base    = float(np.log1p(params["mov_baseline"]))
    ot_discount = float(params["ot_discount"])
    inv_scale = float(params["inv_scale"])

    want = ["game_id", "game_date", "team_id", "is_home", "win", "plus_minus", "num_ot"]
    gl   = game_log[[c for c in want if c in game_log.columns]].sort_values(["game_date", "game_id"])

    home_extra = [c for c in ["plus_minus", "num_ot"] if c in gl.columns]
    home = gl[gl["is_home"]][["game_id", "game_date", "team_id", "win"] + home_extra].copy()
    away = (
        gl[~gl["is_home"]][["game_id", "team_id"]]
        .rename(columns={"team_id": "away_team_id"})
    )

    merged = (
        home.merge(away, on="game_id", how="inner")
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    n = len(merged)

    # .values on a pandas StringDtype column returns an object array, but
    # game_ids.dtype would be StringDtype — not a valid numpy dtype. Convert
    # game_id and game_date to plain object arrays so np.empty works regardless
    # of whether the column uses a numpy or pandas extension backing type.
    game_ids   = np.asarray(merged["game_id"])
    game_dates = np.asarray(merged["game_date"])
    h_ids      = merged["team_id"].values.astype(int)
    a_ids      = merged["away_team_id"].values.astype(int)
    wins_h     = merged["win"].values.astype(float)
    raw_margins = (
        merged["plus_minus"].fillna(0).values.astype(float)
        if "plus_minus" in merged.columns else np.zeros(n, dtype=float)
    )
    n_ots = (
        merged["num_ot"].fillna(0).values.astype(int)
        if "num_ot" in merged.columns else np.zeros(n, dtype=int)
    )

    log.debug(
        "compute_elo: %d games | game_id dtype=%s | game_date dtype=%s | "
        "plus_minus present=%s | num_ot present=%s",
        n,
        game_ids.dtype,
        game_dates.dtype,
        "plus_minus" in merged.columns,
        "num_ot" in merged.columns,
    )

    # Pre-compute log1p(|margin|) for all games up-front
    log1p_margins = np.log1p(np.abs(raw_margins))

    fallback     : dict[int, float] = initial_ratings or {}
    ratings      : dict[int, float] = {}
    home_played  : dict[int, int]   = {}
    home_wins_cnt: dict[int, int]   = {}

    out_gids  = np.empty(n * 2, dtype=object)    # object: safe for any game_id type
    out_dates = np.empty(n * 2, dtype=object)    # object: safe for any date backing
    out_tids  = np.empty(n * 2, dtype=int)
    out_pre   = np.empty(n * 2, dtype=float)
    out_hadv  = np.full(n * 2, np.nan)

    for i in range(n):
        h = int(h_ids[i])
        a = int(a_ids[i])

        r_h = ratings.get(h, fallback.get(h, ELO_INITIAL))
        r_a = ratings.get(a, fallback.get(a, ELO_INITIAL))

        played  = home_played.get(h, 0)
        hw_rate = home_wins_cnt.get(h, 0) / played if played > 0 else 0.5
        hadv    = max(adv_min, adv_base + (hw_rate - 0.5) * adv_scale)

        exp_h   = 1.0 / (1.0 + 10.0 ** ((r_a - r_h - hadv) / inv_scale))
        s_h     = wins_h[i]
        delta_h = (
            k
            * (log1p_margins[i] / log_base)
            * (1.0 / (1.0 + n_ots[i] * ot_discount))
            * (s_h - exp_h)
        )

        j = i * 2
        out_gids[j]  = game_ids[i]
        out_dates[j] = game_dates[i]
        out_tids[j]  = h
        out_pre[j]   = r_h
        out_hadv[j]  = hadv

        out_gids[j + 1]  = game_ids[i]
        out_dates[j + 1] = game_dates[i]
        out_tids[j + 1]  = a
        out_pre[j + 1]   = r_a
        # out_hadv[j+1] stays NaN

        ratings[h] = r_h + delta_h
        ratings[a] = r_a - delta_h  # symmetric: sum of ratings is conserved
        home_played[h] = played + 1
        if s_h == 1.0:
            home_wins_cnt[h] = home_wins_cnt.get(h, 0) + 1

    return pd.DataFrame({
        "game_id":   out_gids,
        "team_id":   out_tids,
        "game_date": out_dates,
        "elo_pre":   out_pre,
        "home_adv":  out_hadv,
    })


# ---------------------------------------------------------------------------
# Vectorised log-loss evaluation
# ---------------------------------------------------------------------------

def _evaluate_log_loss(
    elo: pd.DataFrame,
    home_outcomes: pd.DataFrame,
    inv_scale: float,
    cold_start: int = COLD_START_GAMES,
) -> float:
    """Join home/away Elo per game and return log-loss.

    Fully vectorised — no per-game Python loop.  The first ``cold_start``
    games (chronological) are excluded to let ratings stabilise.
    """
    home_elo = (
        elo.loc[elo["home_adv"].notna(), ["game_id", "game_date", "elo_pre", "home_adv"]]
        .rename(columns={"elo_pre": "home_elo"})
    )
    away_elo = (
        elo.loc[elo["home_adv"].isna(), ["game_id", "elo_pre"]]
        .rename(columns={"elo_pre": "away_elo"})
    )

    games = (
        home_elo
        .merge(away_elo,      on="game_id", how="inner")
        .merge(home_outcomes, on="game_id", how="inner")
        .sort_values("game_date")
        .iloc[cold_start:]
    )

    log.debug(
        "_evaluate_log_loss: %d total games, %d after cold-start skip",
        len(games) + cold_start,
        len(games),
    )

    if len(games) == 0:
        log.warning("_evaluate_log_loss: no games remain after cold-start skip of %d", cold_start)
        return float("nan")

    diff = games["away_elo"].values - games["home_elo"].values - games["home_adv"].values
    p    = np.clip(1.0 / (1.0 + np.power(10.0, diff / inv_scale)), 1e-7, 1.0 - 1e-7)
    y    = games["home_win"].values.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading interim game_log...")
    game_log = read_interim("game_log.parquet")
    log.info("  %d rows, %d cols", game_log.shape[0], game_log.shape[1])
    log.debug("  dtypes:\n%s", game_log.dtypes.to_string())

    prior_final = _load_prior_final_elo()
    if prior_final is None:
        log.info("  No prior-season Elo found — all teams start at 1500 (carryover has no effect)")
    else:
        log.info("  Prior-season Elo loaded: %d teams", len(prior_final))

    # Pre-compute regressed initial ratings for each carryover value
    carryover_map: dict[float, dict[int, float] | None] = {
        c: _make_initial_ratings(prior_final, c) for c in GRID["carryover"]
    }

    # Build home-win lookup once — reused across every combination
    home_outcomes = (
        game_log.loc[game_log["is_home"], ["game_id", "win"]]
        .rename(columns={"win": "home_win"})
        .copy()
    )

    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    total  = len(combos)
    log.info("Starting grid search: %d combinations", total)

    results: list[dict] = []
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        try:
            elo = compute_elo(game_log, params, carryover_map[params["carryover"]])
            ll  = _evaluate_log_loss(elo, home_outcomes, params["inv_scale"])
        except Exception:
            log.exception("Combination %d/%d failed with params=%s", idx, total, params)
            ll = float("nan")
        results.append({**params, "log_loss": ll})

        if idx % 100 == 0 or idx == total:
            valid = [r["log_loss"] for r in results if not np.isnan(r["log_loss"])]
            best  = f"{min(valid):.5f}" if valid else "n/a"
            log.info("%6d / %d  |  latest=%.5f  best=%s", idx, total, ll, best)

    n_failed = sum(np.isnan(r["log_loss"]) for r in results)
    if n_failed:
        log.warning("%d / %d combinations returned NaN log-loss", n_failed, total)

    results_df = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)

    out_path = OUTPUTS_DIR / "models/elo_grid_search_results.parquet"
    write_parquet(results_df, out_path)
    log.info("Saved %d results → %s", len(results_df), out_path.relative_to(PROJECT_ROOT))

    print_table("Top 10 Elo parameter combinations", results_df.head(10))


if __name__ == "__main__":
    main()
