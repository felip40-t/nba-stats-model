# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Constraints

- Do NOT delete any files, ever. If removal is needed, ask the user to do it manually.
- Do NOT read or edit files outside the project root directory. All file operations must stay within the project folder.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

All scripts run from the project root, in order:

```bash
# Stage 1 — fetch raw data from nba_api (~90s for 10 games/team, rate-limited)
python src/data/fetch_games.py

# Stage 2 — clean raw → interim tables (game_log, player_game_log, team_advanced)
python src/data/process.py

# Stage 3 — feature engineering → processed tables (team_features, player_features)
python src/data/features.py

# Launch JupyterLab
jupyter lab

# Run tests
python -m pytest tests/
```

## Architecture

### Data pipeline stages

1. **Ingestion** (`src/data/fetch_games.py`) — Calls `LeagueGameLog` and `BoxScoreTraditionalV3` from `nba_api`. Writes `team_gamelog_raw.parquet` and `boxscore_raw.parquet` to `data/<SEASON>/raw/`.
2. **Processing** (`src/data/process.py`) — Cleans and restructures raw files into three interim tables in `data/<SEASON>/interim/`: `game_log.parquet` (team-per-game with `is_home`, `win`, `is_back_to_back` flags), `player_game_log.parquet` (player-per-game with `minutes_decimal`), `team_advanced.parquet` (aggregated team totals + TS%, 3P rate, FT rate, OREB%).
3. **Feature engineering** (`src/data/features.py`) — Reads interim tables and produces `team_features.parquet` and `player_features.parquet` in `data/<SEASON>/processed/`.

### I/O and path resolution (`src/utils/io.py`)

All path helpers (`read_raw`, `read_interim`, `write_interim`, `write_processed`) resolve under `data/<SEASON>/`. The active season is controlled by `SEASON = "2025"` in `src/utils/io.py`. To add a new season, update this constant — all downstream paths update automatically.

### Key implementation patterns

**No data leakage:** All rolling features use `shift(1).expanding().mean()` per group, so the current game is never included in its own averages. The same pattern applies to `season_win_rate` and `games_played`.

**Column naming:** Raw NBA API columns use both `UPPER_CASE` (game-log) and `camelCase` (box-score). The `_snake()` helper in `process.py` normalises both to `snake_case` before any downstream logic.

**Fatigue metrics** (player-level, in `features.py`):
- `fatigue_decay` — exponential decay load: `Σ minutes_j · e^{-λ(date_i − date_j)}` using `FATIGUE_LAMBDA = 0.2 day⁻¹`. Tune between 0.15 (slow decay) and 0.3 (fast decay).
- `acwr` — Acute:Chronic Workload Ratio: 7-day minutes / (28-day minutes / 4). Values > 1 indicate an acute workload spike.

### Tunable constants

| Constant | File | Default | Purpose |
|---|---|---|---|
| `SEASON` | `src/utils/io.py` | `"2025"` | Active season; controls all data paths |
| `GAMES_PER_TEAM` | `src/data/fetch_games.py` | `10` | Number of games to fetch per team |
| `API_DELAY` | `src/data/fetch_games.py` | `0.6` s | Delay between `nba_api` calls; increase to 1.0+ for bulk back-fills |
| `FATIGUE_LAMBDA` | `src/data/features.py` | `0.2` | Decay rate for exponential fatigue model |

## Key Notes

- `data/` and `outputs/` are gitignored — all data lives locally only.
- Data is stored as Apache Parquet (via `pyarrow`). Use `src/utils/io.py` helpers rather than calling `pd.read_parquet` / `pd.to_parquet` directly.
- Target milestone: working software ready for the 2026 NBA playoffs.
