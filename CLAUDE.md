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
# Stage 1 — fetch schedule + raw data from nba_api (~20 min for full season, rate-limited)
python src/data/fetch_games.py
# To fetch only the schedule: python src/data/fetch_games.py --schedule-only
# Incremental update: python src/data/fetch_games.py --update

# Stage 2 — clean raw → interim tables (game_log, player_game_log, team_advanced)
python src/data/process.py

# Stage 3 — feature engineering → processed tables (team_features, player_features, elo_ratings)
python src/data/features.py

# Stage 4 — rolling day-by-day training simulation
python src/models/train.py

# Stage 5 — evaluate model (metrics + calibration/coefficient plots)
python src/models/evaluate.py

# Launch JupyterLab
jupyter lab

# Run tests
python -m pytest tests/
python -m pytest tests/test_foo.py::test_bar  # single test
```

## Architecture

### Data pipeline stages

1. **Ingestion** (`src/data/fetch_games.py`) — Calls `ScheduleLeagueV2`, `LeagueGameLog`, and `BoxScoreTraditionalV3` from `nba_api`. Writes `schedule.parquet`, `team_gamelog_raw.parquet`, and `boxscore_raw.parquet` to `data/<SEASON>/raw/`. Supports `--schedule-only`, `--update` (incremental), and `--playoffs` flags.
2. **Processing** (`src/data/process.py`) — Cleans and restructures raw files into three interim tables in `data/<SEASON>/interim/`: `game_log.parquet` (team-per-game with `is_home`, `win`, `is_back_to_back` flags), `player_game_log.parquet` (player-per-game with `minutes_decimal`), `team_advanced.parquet` (aggregated team totals + TS%, 3P rate, FT rate, OREB%).
3. **Feature engineering** (`src/data/features.py`) — Reads interim tables and produces `team_features.parquet`, `player_features.parquet`, and `elo_ratings.parquet` in `data/<SEASON>/processed/`. Also computes per-team Elo ratings (`elo_pre`, `elo_post`, `home_adv`) and team-level fatigue aggregates (`team_fatigue`, `team_acwr`). Exposes `compute_features_from_data()` for in-memory computation with an optional `cutoff_date`; disk writes only happen when run directly (no cutoff = end-of-season snapshot).
4. **Training** (`src/models/train.py`) — Rolling day-by-day simulation: for each game date `d`, recomputes features in memory for all games ≤ `d`, trains on games before `d`, predicts games on `d`. Saves `outputs/models/win_probability_logreg.joblib` (final model, full season) and `outputs/models/rolling_predictions.parquet` (each game predicted once using only prior data).
5. **Evaluation** (`src/models/evaluate.py`) — Loads saved model and rolling predictions; prints log-loss, Brier score, and accuracy; saves calibration curve and feature-coefficient plots to `outputs/figures/`.

### I/O and path resolution (`src/utils/io.py`)

All path helpers (`read_raw`, `write_raw`, `read_interim`, `write_interim`, `read_processed`, `write_processed`, `read_schedule`) resolve under `data/<SEASON>/`. The active season is controlled by `SEASON = "2025"` in `src/utils/io.py`. To add a new season, update this constant — all downstream paths update automatically. `season_api(season)` converts the project's ending-year label (e.g. `"2025"`) to the `nba_api` hyphenated format (`"2024-25"`). Model outputs go to `outputs/models/` and figures to `outputs/figures/` (also gitignored).

### Key implementation patterns

**No data leakage:** All rolling features use `shift(1).expanding().mean()` per group, so the current game is never included in its own averages. The same pattern applies to `season_win_rate` and `games_played`.

**Column naming:** Raw NBA API columns use both `UPPER_CASE` (game-log) and `camelCase` (box-score). The `_snake()` helper in `process.py` normalises both to `snake_case` before any downstream logic.

**Fatigue metrics** (player-level, in `features.py`):
- `fatigue_decay` — exponential decay load: `Σ minutes_j · e^{-λ(date_i − date_j)}` using `FATIGUE_LAMBDA = 0.2 day⁻¹`. Tune between 0.15 (slow decay) and 0.3 (fast decay).
- `acwr` — Acute:Chronic Workload Ratio: 7-day minutes / (28-day minutes / 4). Values > 1 indicate an acute workload spike.

### Model features (`src/models/train.py`)

The logistic regression trains on **home-minus-away deltas** for: `elo_delta`, `win_rate_delta`, `pts_delta`, `fg_pct_delta`, `fatigue_delta`, `acwr_delta`, plus `home_adv` (team-specific Elo home-court bonus). Only `elo_pre` (not `elo_post`) is used during training. There is no static train/test split — instead, each game is predicted exactly once using only data from before that game date (rolling simulation).

### Tunable constants

| Constant | File | Default | Purpose |
|---|---|---|---|
| `SEASON` | `src/utils/io.py` | `"2025"` | Active season; controls all data paths |
| `API_DELAY` | `src/data/fetch_games.py` | `0.6` s | Delay between `nba_api` calls; increase to 1.0+ for bulk back-fills |
| `FATIGUE_LAMBDA` | `src/data/features.py` | `0.2` | Decay rate for exponential fatigue model |
| `ELO_INITIAL` | `src/data/features.py` | `1500.0` | Starting Elo rating for all teams |
| `ELO_K` | `src/data/features.py` | `20.0` | Elo K-factor (update magnitude per game) |
| `ELO_HOME_ADV_BASE` | `src/data/features.py` | `100.0` | Elo home-court bonus for a .500 home record |

## Key Notes

- `data/` and `outputs/` are gitignored — all data lives locally only.
- Data is stored as Apache Parquet (via `pyarrow`). Use `src/utils/io.py` helpers rather than calling `pd.read_parquet` / `pd.to_parquet` directly.
- Target milestone: working software ready for the 2026 NBA playoffs.
