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

## Commands

```bash
# Run the full test suite
python -m pytest tests/

# Run a single test
python -m pytest tests/test_features.py::test_compute_elo_sum_conservation

# Lint
ruff check src/

# Format
ruff format src/

# Launch JupyterLab
jupyter lab
```

## Running the Pipeline

All scripts run from the project root, in order:

```bash
# Stage 1 — fetch schedule + raw data from nba_api (~20 min for full season, rate-limited)
python src/data/fetch_games.py
# To fetch only the schedule: python src/data/fetch_games.py --schedule-only
# Incremental update: python src/data/fetch_games.py --update
# Playoffs: python src/data/fetch_games.py --playoffs [--update]

# Stage 2 — clean raw → interim tables (game_log, player_game_log, team_advanced)
python src/data/process.py
# Playoffs: python src/data/process.py --playoffs

# Stage 3 — feature engineering → processed tables (team_features, player_features, elo_ratings)
python src/data/features.py
# Playoffs: python src/data/features.py --playoffs

# Stage 4 — rolling day-by-day training simulation
python src/models/train.py

# Stage 5 — evaluate model (metrics + calibration/coefficient plots)
python src/models/evaluate.py
```

**Common flags:**

```bash
# Full playoffs pipeline (run after completing the regular-season pipeline above)
python src/data/fetch_games.py --playoffs [--update]
python src/data/process.py --playoffs
python src/data/features.py --playoffs

# Train a playoffs-ready model on all regular-season data
python src/models/train.py --playoffs

# Raise the cold-start threshold (default 50)
python src/models/train.py --min-train-games 100
```

## Architecture

### Data pipeline stages

1. **Ingestion** (`src/data/fetch_games.py`) — Calls `ScheduleLeagueV2`, `LeagueGameLog`, and `BoxScoreTraditionalV3` from `nba_api`. Writes `schedule.parquet`, `team_gamelog_raw.parquet`, and `boxscore_raw.parquet` to `data/<SEASON>/raw/`. Supports `--schedule-only`, `--update` (incremental), and `--playoffs` flags. For playoffs, output files carry a `_playoffs` suffix (e.g. `boxscore_raw_playoffs.parquet`). Checkpoints box-score parquet to disk every `CHECKPOINT_EVERY` games so interrupted runs resume via `--update`. Uses exponential-backoff retry (up to `MAX_RETRIES` attempts) and a periodic cooldown pause every `COOLDOWN_EVERY` games.
2. **Processing** (`src/data/process.py`) — Cleans and restructures raw files into three interim tables in `data/<SEASON>/interim/`: `game_log.parquet` (team-per-game with `is_home`, `win`, `is_back_to_back`, `num_ot` flags), `player_game_log.parquet` (player-per-game with `minutes_decimal`), `team_advanced.parquet` (aggregated team totals + TS%, 3P rate, FT rate, OREB%). `num_ot` is derived from summed player-minutes: regulation = 240 per team, each OT adds 25. Pass `--playoffs` to process the `_playoffs` raw files and write `_playoffs` interim files.
3. **Feature engineering** (`src/data/features.py`) — Reads interim tables and produces `team_features.parquet`, `player_features.parquet`, and `elo_ratings.parquet` in `data/<SEASON>/processed/`. Also computes per-team Elo ratings (`elo_pre`, `elo_post`, `home_adv`) and team-level fatigue aggregates (`team_fatigue`, `team_acwr`). Exposes `compute_features_from_data()` for in-memory computation with an optional `cutoff_date`; disk writes only happen when run directly. Pass `--playoffs` to load `*_playoffs.parquet` interim files alongside the regular-season tables; regular-season data is always included so Elo ratings carry over and rolling averages are not cold-started.
4. **Training** (`src/models/train.py`) — Loads the full processed feature snapshot from disk once, then for each game date `d`: trains on all complete rows with `game_date < d`, predicts all games on `d`. Feature leakage is prevented by the `shift(1).expanding()` encoding already baked into the parquet — no per-iteration recompute is needed. Dates with fewer than `min_train_games` (default 50) complete rows are skipped (cold-start). Saves `outputs/models/win_probability_logreg.joblib` and `outputs/models/rolling_predictions.parquet`.
5. **Evaluation** (`src/models/evaluate.py`) — Loads saved model and rolling predictions; prints log-loss, Brier score, and accuracy; saves nine diagnostic plots to `outputs/figures/`.

### I/O and path resolution (`src/utils/io.py`)

All path helpers (`read_raw`, `write_raw`, `read_interim`, `write_interim`, `read_processed`, `write_processed`, `read_schedule`) resolve under `data/<SEASON>/`. The active season is controlled by `SEASON = "2025"` in `src/utils/io.py`. To add a new season, update this constant — all downstream paths update automatically. `season_api(season)` converts the project's ending-year label (e.g. `"2025"`) to the `nba_api` hyphenated format (`"2024-25"`). Model outputs go to `outputs/models/` and figures to `outputs/figures/` (also gitignored).

Always use the `src/utils/io.py` helpers rather than calling `pd.read_parquet` / `pd.to_parquet` directly.

### Key implementation patterns

**No data leakage:** All rolling features use `shift(1).expanding().mean()` per group, so the current game is never included in its own averages. The same pattern applies to `season_win_rate` and `games_played`. Because this encoding is baked into the processed parquet, `train.py` can safely slice by `game_date < d` without re-running features per iteration.

**Column naming:** Raw NBA API columns use both `UPPER_CASE` (game-log) and `camelCase` (box-score). The `_snake()` helper in `process.py` normalises both to `snake_case` before any downstream logic.

**Fatigue metrics** (player-level, in `features.py`):
- `fatigue_decay` — exponential decay load: `Σ minutes_j · e^{-λ(date_i − date_j)}` using `FATIGUE_LAMBDA = 0.2 day⁻¹`.
- `acwr` — Acute:Chronic Workload Ratio: 7-day minutes / (28-day minutes / 4). Values > 1 indicate an acute workload spike.

**Team-level fatigue aggregation** (`build_team_player_features`): player fatigue metrics are aggregated to the team-game level as minutes-weighted averages, using each player's `season_avg_minutes_decimal` as the weight (shift-by-1, so DNP/debut players are excluded).

**Elo home-court advantage:** Only the home team's row carries `home_adv` (the away team's field is `None` in the raw Elo output). The value is team-specific: `max(ELO_HOME_ADV_MIN, ELO_HOME_ADV_BASE + (home_win_rate − 0.5) × ELO_HOME_ADV_SCALE)`.

**CLI output:** `src/utils/display.py` exports `print_table(title, df)`, used by `features.py` to print interim DataFrames at the end of each stage when run directly.

### Model features (`src/models/train.py`)

`build_game_rows()` pivots team features into one-row-per-game format using the schedule to identify home/away sides. `compute_deltas()` adds model input columns as home-minus-away differences. `drop_missing()` removes rows where any feature is NaN (expected for early-season games before rolling windows have enough history). The logistic regression trains on **home-minus-away deltas** for: `elo_delta`, `win_rate_delta`, `pts_delta`, `fg_pct_delta`, `fatigue_delta`, `acwr_delta`, `home_adv`, plus box-score deltas (`ast`, `reb`, `oreb`, `blk`, `stl`, `tov`, `pf`, `fta`, `ft_pct`, `plus_minus`). Only `elo_pre` (not `elo_post`) is used.

### Tests (`tests/`)

`conftest.py` adds the project root to `sys.path` so `src.*` imports work. All tests construct minimal DataFrames directly — there is no test data on disk and no I/O. `pyproject.toml` configures pytest (`testpaths = ["tests"]`, `addopts = "-ra -q"`) and ruff (line-length 100, `target-version = "py310"`).

### Tunable constants

| Constant | File | Default | Purpose |
|---|---|---|---|
| `SEASON` | `src/utils/io.py` | `"2025"` | Active season; controls all data paths |
| `API_DELAY` | `src/data/fetch_games.py` | `0.6` s | Delay between `nba_api` calls for small runs |
| `BULK_API_DELAY` | `src/data/fetch_games.py` | `1.0` s | Delay used when fetching >100 new games |
| `CHECKPOINT_EVERY` | `src/data/fetch_games.py` | `100` | Games between intermediate box-score saves |
| `COOLDOWN_EVERY` | `src/data/fetch_games.py` | `200` | Games between long cooldown pauses |
| `COOLDOWN_SECONDS` | `src/data/fetch_games.py` | `30.0` s | Duration of each cooldown pause |
| `FATIGUE_LAMBDA` | `src/data/features.py` | `0.2` | Decay rate for exponential fatigue model |
| `ELO_INITIAL` | `src/data/features.py` | `1500.0` | Starting Elo rating for all teams |
| `ELO_K` | `src/data/features.py` | `20.0` | Elo K-factor (update magnitude per game) |
| `ELO_HOME_ADV_BASE` | `src/data/features.py` | `100.0` | Elo home-court bonus for a .500 home record |
| `ELO_MOV_BASELINE` | `src/data/features.py` | `8.0` | Point margin yielding a MOV multiplier of 1.0 (≈ median NBA win margin) |
| `ELO_OT_DISCOUNT` | `src/data/features.py` | `0.5` | K reduction per OT period: `ot_factor = 1 / (1 + num_ot × discount)` |
| `min_train_games` | `src/models/train.py` | `50` | Minimum training rows before the rolling simulation starts predicting |

## Key Notes

- `data/` and `outputs/` are gitignored — all data lives locally only.
- Target milestone: working software ready for the 2026 NBA playoffs.
