"""
fetch_games.py — Step 1 of the ingestion pipeline.

Fetches the first game of the 2024-25 NBA season and writes raw data
to data/raw/ as Parquet files. Designed to be run from the project root:

    python src/data/fetch_games.py
"""

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nba_api.stats.endpoints import BoxScoreTraditionalV3, LeagueGameLog


# Project root is two levels up from this file (src/data/fetch_games.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# --- API calls -----------------------------------------------------------

def fetch_team_gamelog(season: str = "2024-25") -> pd.DataFrame:
    """
    Fetch every team-game entry for the given season via LeagueGameLog.

    nba_api sends a browser-like User-Agent header automatically; no auth
    needed. The endpoint returns one row per team per game, so each game
    appears twice (once per side).
    """
    log = LeagueGameLog(season=season, season_type_all_star="Regular Season")
    df = log.get_data_frames()[0]
    return df


def fetch_boxscore(game_id: str) -> dict[str, pd.DataFrame]:
    """
    Fetch the traditional box score for a single game.

    BoxScoreTraditionalV2 returns multiple result sets; we keep:
      - PlayerStats  (one row per player)
      - TeamStats    (one row per team, two rows total)
    """
    box = BoxScoreTraditionalV3(game_id=game_id)
    frames = box.get_data_frames()
    return {
        "player_stats": frames[0],
        "team_stats": frames[1],
    }


# --- Isolation logic -----------------------------------------------------

def get_first_game(gamelog_df: pd.DataFrame) -> tuple[str, str]:
    """
    Return (game_id, game_date) for the earliest game in the log.

    GAME_DATE comes back as a string ('2024-10-22'), so lexicographic
    sort is fine for finding the minimum date.
    """
    earliest_date = gamelog_df["GAME_DATE"].min()
    first_game_row = gamelog_df[gamelog_df["GAME_DATE"] == earliest_date].iloc[0]
    return first_game_row["GAME_ID"], earliest_date


# --- File I/O ------------------------------------------------------------

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


# --- Orchestration -------------------------------------------------------

def main() -> None:
    print("Fetching team game log for 2024-25 season...")
    gamelog_df = fetch_team_gamelog(season="2024-25")

    game_id, game_date = get_first_game(gamelog_df)
    print(f"First game of season: {game_date}  (GAME_ID={game_id})")

    # Isolate the two rows for that game before writing
    first_game_rows = gamelog_df[gamelog_df["GAME_ID"] == game_id].copy()

    gamelog_path = RAW_DIR / "team_gamelog_raw.parquet"
    write_parquet(first_game_rows, gamelog_path)
    print(f"  -> saved team gamelog to {gamelog_path.relative_to(PROJECT_ROOT)}")

    # nba_api's stats.nba.com backend enforces a soft rate limit;
    # 0.6 s is enough to avoid 429s during normal single-script runs.
    time.sleep(0.6)

    print("Fetching box score...")
    boxscore = fetch_boxscore(game_id)

    boxscore_path = RAW_DIR / "boxscore_raw.parquet"
    # Combine player and team stats into one file with a 'stat_type' column
    boxscore["player_stats"]["stat_type"] = "player"
    boxscore["team_stats"]["stat_type"] = "team"
    combined = pd.concat(
        [boxscore["player_stats"], boxscore["team_stats"]], ignore_index=True
    )
    write_parquet(combined, boxscore_path)
    print(f"  -> saved box score to   {boxscore_path.relative_to(PROJECT_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
