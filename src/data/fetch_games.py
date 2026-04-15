"""
fetch_games.py — Step 1 of the ingestion pipeline.

Fetches the first ``GAMES_PER_TEAM`` games of each team for the 2024-25
NBA season and writes raw data to ``data/2025/raw/`` as Parquet files.

Output files
------------
team_gamelog_raw.parquet
    All team-game rows (both sides of each game) for the selected games.
boxscore_raw.parquet
    Combined player + team box-score rows for every selected game,
    tagged with a ``stat_type`` column (``"player"`` / ``"team"``).

Run from the project root::

    python src/data/fetch_games.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nba_api.stats.endpoints import BoxScoreTraditionalV3, LeagueGameLog  # noqa: E402

from src.utils.io import PROJECT_ROOT, RAW_DIR, write_parquet  # noqa: E402

# Number of games to collect per team (from the start of the season).
GAMES_PER_TEAM: int = 10

# Minimum delay between NBA Stats API calls (seconds).
# stats.nba.com enforces a soft rate limit; 0.6 s is sufficient for single-
# script runs.  Increase to 1.0+ for batch back-fills to be safe.
API_DELAY: float = 0.6


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_team_gamelog(season: str = "2024-25") -> pd.DataFrame:
    """Return the full team game-log for *season* (regular season only).

    Each game appears twice — once per participating team.
    """
    log = LeagueGameLog(season=season, season_type_all_star="Regular Season")
    return log.get_data_frames()[0]


def get_first_n_game_ids(gamelog_df: pd.DataFrame, n: int) -> list[str]:
    """Return unique game IDs covering the first *n* games of every team.

    Sorts each team's appearances chronologically and takes the first *n*,
    then returns the union of distinct ``GAME_ID`` values.  Because each
    game is shared by two teams the total number of unique IDs is
    typically around ``(n * 30) / 2``.
    """
    df = gamelog_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    first_n = df.groupby("TEAM_ID").head(n)
    return sorted(first_n["GAME_ID"].unique().tolist())


def fetch_boxscore(game_id: str) -> dict[str, pd.DataFrame]:
    """Fetch the traditional box score for a single game.

    Returns a dict with keys ``"player_stats"`` and ``"team_stats"``.
    """
    box = BoxScoreTraditionalV3(game_id=game_id)
    frames = box.get_data_frames()
    return {"player_stats": frames[0], "team_stats": frames[1]}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    season = "2024-25"
    print(f"Fetching team game log for {season} season...")
    gamelog_df = fetch_team_gamelog(season=season)
    n_teams = gamelog_df["TEAM_ID"].nunique()
    print(f"  Full log: {len(gamelog_df)} rows across {n_teams} teams")

    game_ids = get_first_n_game_ids(gamelog_df, GAMES_PER_TEAM)
    print(f"  First {GAMES_PER_TEAM} games per team → {len(game_ids)} unique games")

    # Write the gamelog rows for the selected games (both team sides)
    selected_rows = gamelog_df[gamelog_df["GAME_ID"].isin(game_ids)].copy()
    gamelog_path = RAW_DIR / "team_gamelog_raw.parquet"
    write_parquet(selected_rows, gamelog_path)
    print(f"  -> saved team gamelog ({len(selected_rows)} rows) "
          f"to {gamelog_path.relative_to(PROJECT_ROOT)}")

    # Fetch box scores for every unique game
    print(f"\nFetching {len(game_ids)} box scores "
          f"(~{len(game_ids) * API_DELAY:.0f}s with rate limiting)...")
    all_frames: list[pd.DataFrame] = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"  [{i:3d}/{len(game_ids)}] {game_id}", end="\r", flush=True)
        boxscore = fetch_boxscore(game_id)
        boxscore["player_stats"]["stat_type"] = "player"
        boxscore["team_stats"]["stat_type"] = "team"
        combined = pd.concat(
            [boxscore["player_stats"], boxscore["team_stats"]],
            ignore_index=True,
        )
        all_frames.append(combined)
        time.sleep(API_DELAY)

    boxscore_df = pd.concat(all_frames, ignore_index=True)
    boxscore_path = RAW_DIR / "boxscore_raw.parquet"
    write_parquet(boxscore_df, boxscore_path)
    print(f"\n  -> saved box scores ({len(boxscore_df)} rows) "
          f"to {boxscore_path.relative_to(PROJECT_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
