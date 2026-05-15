"""
fetch_games.py — Stage 1 of the ingestion pipeline.

Fetches the season schedule, team game-logs, and player/team box scores from
``nba_api`` for the active season (controlled by ``SEASON`` in
``src/utils/io.py``) and writes raw Parquet files to ``data/<SEASON>/raw/``.

Output files (regular season)
-----------------------------
schedule.parquet
    One row per game (including unplayed future games) with home/away team IDs
    and abbreviations.  Written by every full run and by ``--schedule-only``.
team_gamelog_raw.parquet
    All team-game rows (both sides of each game) for the season.
boxscore_raw.parquet
    Combined player + team box-score rows for every game, tagged with a
    ``stat_type`` column (``"player"`` / ``"team"``).

For playoffs, the gamelog and boxscore files are written with a ``_playoffs``
suffix (e.g. ``boxscore_raw_playoffs.parquet``).

Modes
-----
* **Full fetch** (default): fetch schedule + every game in the active season's
  gamelog, overwriting any existing raw files.
* **Schedule only** (``--schedule-only``): fetch and save just the schedule;
  skip game-log and box-score fetching entirely.
* **Incremental update** (``--update``): load the existing raw files, fetch
  only game IDs not yet on disk, then append.  Use this daily to pick up
  newly-played games without re-fetching the back-catalog.
* **Playoffs** (``--playoffs``): same behaviour as above but queries the
  ``"Playoffs"`` season type and writes to the ``_playoffs`` files.

Checkpointing
-------------
For long back-fills (~1,200 games × ~1s each = ~20 min), the box-score
parquet is re-written every ``CHECKPOINT_EVERY`` games so a crash mid-run
doesn't lose progress — resumable via ``--update`` on the next invocation.

Examples
--------
    # fetch schedule only (no box scores):
    python src/data/fetch_games.py --schedule-only

    # full regular-season back-fill for the active season (~20 min):
    python src/data/fetch_games.py

    # pick up newly-played games since the last run:
    python src/data/fetch_games.py --update

    # playoffs equivalents:
    python src/data/fetch_games.py --playoffs
    python src/data/fetch_games.py --playoffs --update
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_HERE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_HERE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERE_PROJECT_ROOT))

from nba_api.stats.endpoints import BoxScoreTraditionalV3, LeagueGameLog, ScheduleLeagueV2  # noqa: E402

from src.utils.io import (  # noqa: E402
    PROJECT_ROOT,
    RAW_DIR,
    SEASON,
    configure_logging,
    read_parquet,
    season_api,
    write_parquet,
    write_raw,
)

# Default delay between NBA Stats API calls.  stats.nba.com enforces a soft
# rate limit; 0.6 s is fine for small runs but we bump to 1.0 s for bulk
# back-fills (>100 games) to be safe over long sessions.
DEFAULT_API_DELAY: float = 0.6
BULK_API_DELAY: float = 1.0
BULK_THRESHOLD: int = 100

# Write an intermediate snapshot of the box-score parquet every N games so a
# crash mid-fetch doesn't lose progress (resume with --update).
CHECKPOINT_EVERY: int = 100

# Retry / cooldown settings for long back-fills.
MAX_RETRIES: int = 3          # attempts per game before giving up
RETRY_BACKOFF_BASE: float = 5.0  # seconds; doubles each attempt (5 → 10 → 20)
COOLDOWN_EVERY: int = 200     # games between long pauses
COOLDOWN_SECONDS: float = 30.0   # pause duration to let the rate-limit window reset

log = configure_logging("fetch_games")


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def fetch_schedule(season: str = SEASON) -> pd.DataFrame:
    """Fetch the full regular-season schedule and return a tidy DataFrame.

    Includes future unplayed games (null scores) so the schedule can be used
    to frame predictions.  One row per game with columns ``game_id``,
    ``game_date``, ``home_team_id``, ``away_team_id``,
    ``home_team_abbreviation``, ``away_team_abbreviation``.
    """
    api_season = season_api(season)
    print(f"  Fetching ScheduleLeagueV2 for season {api_season} ...")
    sched = ScheduleLeagueV2(season=api_season)
    raw = sched.get_data_frames()[0]

    keep = {
        "gameId": "game_id",
        "gameDate": "game_date",
        "homeTeam_teamId": "home_team_id",
        "awayTeam_teamId": "away_team_id",
        "homeTeam_teamTricode": "home_team_abbreviation",
        "awayTeam_teamTricode": "away_team_abbreviation",
    }
    df = raw[list(keep.keys())].rename(columns=keep).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_team_id"] = df["home_team_id"].astype("Int64")
    df["away_team_id"] = df["away_team_id"].astype("Int64")
    return df.sort_values("game_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_team_gamelog(
    season: str,
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """Return the full team game-log for *season* and *season_type*.

    Each game appears twice — once per participating team.
    """
    log = LeagueGameLog(season=season_api(season), season_type_all_star=season_type)
    return log.get_data_frames()[0]


def get_game_ids(gamelog_df: pd.DataFrame) -> list[str]:
    """Return all unique game IDs from the gamelog, sorted."""
    return sorted(gamelog_df["GAME_ID"].unique().tolist())


def fetch_boxscore(game_id: str) -> dict[str, pd.DataFrame]:
    """Fetch the traditional box score for a single game.

    Returns a dict with keys ``"player_stats"`` and ``"team_stats"``.
    """
    box = BoxScoreTraditionalV3(game_id=game_id)
    frames = box.get_data_frames()
    return {"player_stats": frames[0], "team_stats": frames[1]}


# ---------------------------------------------------------------------------
# Filename + existing-data helpers
# ---------------------------------------------------------------------------

def _raw_filename(base: str, playoffs: bool) -> str:
    """Insert ``_playoffs`` before the extension when ``playoffs`` is True."""
    if not playoffs:
        return base
    assert base.endswith(".parquet")
    return base[: -len(".parquet")] + "_playoffs.parquet"


def _existing_game_ids(boxscore_path: Path) -> set[str]:
    """Return the set of ``gameId`` values already present on disk.

    Reads only the ``gameId`` column via pyarrow so we don't pay to
    materialise the full box-score DataFrame just to check membership.
    """
    if not boxscore_path.exists():
        return set()
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(boxscore_path)
        col = next((c for c in ("gameId", "game_id", "GAME_ID") if c in schema.names), None)
        if col is None:
            return set()
        table = pq.read_table(boxscore_path, columns=[col])
        return {str(v) for v in table.column(col).to_pylist() if v is not None}
    except Exception:  # fall back to full read on unexpected schema/engine issues
        df = read_parquet(boxscore_path)
        for col in ("gameId", "game_id", "GAME_ID"):
            if col in df.columns:
                return set(df[col].astype(str).unique())
        return set()


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def run_schedule_only(season: str) -> None:
    """Fetch and save the season schedule; skip all game-log/box-score work."""
    print("Fetching season schedule...")
    schedule = fetch_schedule(season)
    dest = write_raw(schedule, "schedule.parquet")
    print(
        f"  -> saved schedule ({len(schedule)} games) "
        f"to {dest.relative_to(PROJECT_ROOT)}"
    )
    print("Done.")


def run(
    season: str,
    playoffs: bool = False,
    update: bool = False,
    refresh_schedule: bool = False,
    api_delay: float | None = None,
) -> None:
    """Execute a fetch for the given season / mode.

    Always fetches the schedule first (writes ``schedule.parquet``), then
    fetches game-log and box scores.

    Parameters
    ----------
    season:
        Season directory label (e.g. ``"2025"``).
    playoffs:
        If True, fetch the Playoffs season type and write ``_playoffs`` files.
    update:
        If True, keep existing on-disk files and only fetch game IDs not yet
        present.
    api_delay:
        Seconds between API calls.  Defaults to :data:`DEFAULT_API_DELAY`
        (or :data:`BULK_API_DELAY` when fetching >100 new games).
    """
    schedule_path = RAW_DIR / "schedule.parquet"
    if schedule_path.exists() and not refresh_schedule:
        print(f"Schedule already on disk ({schedule_path.relative_to(PROJECT_ROOT)}) — skipping fetch.")
    else:
        if refresh_schedule and schedule_path.exists():
            print("Refreshing schedule (overwriting on-disk copy) ...")
        run_schedule_only(season)
    print()

    season_type = "Playoffs" if playoffs else "Regular Season"
    gamelog_path = RAW_DIR / _raw_filename("team_gamelog_raw.parquet", playoffs)
    boxscore_path = RAW_DIR / _raw_filename("boxscore_raw.parquet", playoffs)

    mode_str = " (incremental update)" if update else ""
    print(f"Fetching {season} {season_type}{mode_str}...")

    # --- Step 1: team gamelog (1 API call) -----------------------------------
    gamelog_df = fetch_team_gamelog(season=season, season_type=season_type)
    n_teams = gamelog_df["TEAM_ID"].nunique()
    print(f"  Full log: {len(gamelog_df)} rows across {n_teams} teams")

    # --- Step 2: figure out which box scores we still need -------------------
    existing_ids: set[str] = _existing_game_ids(boxscore_path) if update else set()
    target_game_ids = get_game_ids(gamelog_df)
    new_game_ids = [g for g in target_game_ids if g not in existing_ids]
    print(f"  Target games: {len(target_game_ids)}")
    log.info("New game IDs to fetch: %d", len(new_game_ids))
    log.info("Skipped (already on disk): %d game IDs", len(existing_ids))

    # Merge new gamelog rows with existing ones so --update runs accumulate.
    if update and gamelog_path.exists():
        existing_log = read_parquet(gamelog_path)
        gamelog_df = (
            pd.concat([existing_log, gamelog_df], ignore_index=True)
            .drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
        )
    write_parquet(gamelog_df, gamelog_path)
    print(
        f"  -> saved team gamelog ({len(gamelog_df)} rows) "
        f"to {gamelog_path.relative_to(PROJECT_ROOT)}"
    )

    if update:
        print(f"  Already on disk : {len(existing_ids)}")
        print(f"  New to fetch    : {len(new_game_ids)}")

    if not new_game_ids:
        print("\nNothing to fetch — already up to date.")
        return

    # --- Step 3: fetch box scores -------------------------------------------
    delay = api_delay if api_delay is not None else (
        BULK_API_DELAY if len(new_game_ids) > BULK_THRESHOLD else DEFAULT_API_DELAY
    )
    est_sec = len(new_game_ids) * delay
    print(
        f"\nFetching {len(new_game_ids)} box scores "
        f"(~{est_sec / 60:.1f} min with {delay}s delay)..."
    )

    # Accumulate new game frames; seed with existing data so every checkpoint
    # write is a complete, resumable snapshot.
    new_frames: list[pd.DataFrame] = []
    existing_boxscores: pd.DataFrame | None = (
        read_parquet(boxscore_path) if (update and boxscore_path.exists()) else None
    )

    def _checkpoint() -> None:
        if not new_frames:
            return
        parts = ([existing_boxscores] if existing_boxscores is not None else []) + new_frames
        write_parquet(pd.concat(parts, ignore_index=True), boxscore_path)

    failed: list[str] = []
    for i, game_id in enumerate(new_game_ids, 1):
        print(f"  [{i:4d}/{len(new_game_ids)}] {game_id}", end="\r", flush=True)

        # Exponential-backoff retry so transient errors don't permanently skip a game.
        boxscore = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                boxscore = fetch_boxscore(game_id)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"\n  WARN: {game_id} attempt {attempt} failed ({exc}); retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    print(f"\n  WARN: {game_id} failed after {MAX_RETRIES} attempts — skipping.")
                    failed.append(game_id)

        if boxscore is None:
            time.sleep(delay)
            continue

        boxscore["player_stats"]["stat_type"] = "player"
        boxscore["team_stats"]["stat_type"] = "team"
        combined = pd.concat(
            [boxscore["player_stats"], boxscore["team_stats"]],
            ignore_index=True,
        )
        new_frames.append(combined)

        if i % CHECKPOINT_EVERY == 0:
            _checkpoint()

        # Periodic cooldown to reset the server-side rate-limit window.
        if i % COOLDOWN_EVERY == 0:
            print(f"\n  Cooldown: pausing {COOLDOWN_SECONDS:.0f}s after {i} games...")
            time.sleep(COOLDOWN_SECONDS)
        else:
            time.sleep(delay)

    # Final snapshot
    _checkpoint()

    if new_frames:
        total_rows = (len(existing_boxscores) if existing_boxscores is not None else 0) + sum(len(f) for f in new_frames)
        print(
            f"\n  -> saved box scores ({total_rows} rows total, "
            f"{len(new_game_ids) - len(failed)} new games) "
            f"to {boxscore_path.relative_to(PROJECT_ROOT)}"
        )
        log.info("Total box score rows written: %d", total_rows)

    if failed:
        preview = ", ".join(failed[:5]) + (" ..." if len(failed) > 5 else "")
        print(f"\n  {len(failed)} game(s) failed: {preview}")
        print("  Re-run with --update to retry.")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch NBA game data for the active season."
    )
    p.add_argument(
        "--schedule-only",
        action="store_true",
        help="Fetch and save only the season schedule; skip game-log and box scores.",
    )
    p.add_argument(
        "--playoffs",
        action="store_true",
        help="Fetch playoffs instead of regular season; writes *_playoffs.parquet.",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Incremental: only fetch games not already in the raw box-score file.",
    )
    p.add_argument(
        "--refresh-schedule",
        action="store_true",
        help="Re-fetch the schedule even if one is already on disk (e.g. to pick up makeup games).",
    )
    p.add_argument(
        "--api-delay",
        type=float,
        default=None,
        metavar="SEC",
        help=f"Seconds between API calls (default: {DEFAULT_API_DELAY}; "
             f"{BULK_API_DELAY} when >{BULK_THRESHOLD} new games).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.schedule_only:
        run_schedule_only(season=SEASON)
        return
    run(
        season=SEASON,
        playoffs=args.playoffs,
        update=args.update,
        refresh_schedule=args.refresh_schedule,
        api_delay=args.api_delay,
    )


if __name__ == "__main__":
    main()
