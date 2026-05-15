# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Constraints

- Do NOT delete any files, ever. Ask the user to do it manually.
- Do NOT read or edit files outside the project root directory.

## Environment Setup

```bash
source .venv/bin/activate && pip install -r requirements.txt
```

## Common Commands

The `Makefile` uses `.venv/bin/python` directly — no need to activate the venv first.

```bash
make pipeline                    # full logreg pipeline (fetch → evaluate)
make pipeline-xgboost            # same with XGBoost
make pipeline-playoffs           # playoffs pipeline (fetch → train, no evaluate)
make pipeline-xgboost-playoffs

make fetch / fetch-update / fetch-schedule / fetch-playoffs / fetch-playoffs-update
make process / process-playoffs
make features / features-playoffs
make train / train-xgboost / train-playoffs / train-xgboost-playoffs
make evaluate / evaluate-xgboost
make elo-grid-search / xgboost-grid-search
make test / lint / format
```

## Running the Pipeline

All scripts run from the project root in order:

```bash
python src/data/fetch_games.py      # Stage 1: fetch raw data (~20 min, rate-limited)
python src/data/process.py          # Stage 2: raw → interim tables
python src/data/features.py         # Stage 3: feature engineering → processed tables
python src/models/train.py          # Stage 4: rolling day-by-day train (logreg)
python src/models/train.py --model xgboost
python src/models/evaluate.py       # Stage 5: metrics + plots
python src/models/evaluate.py --model xgboost
```

**Key flags:** `--playoffs`, `--update` (incremental), `--schedule-only`, `--min-train-games N`

## Architecture

**Data flow:** `data/<SEASON>/raw/` → `interim/` → `processed/` → `outputs/models/` + `outputs/figures/`

**Stages:**
1. `fetch_games.py` — calls `nba_api` (`ScheduleLeagueV2`, `LeagueGameLog`, `BoxScoreTraditionalV3`); checkpoints every 100 games; exponential-backoff retry.
2. `process.py` — produces `game_log.parquet`, `player_game_log.parquet`, `team_advanced.parquet` in `interim/`. `num_ot` derived from player-minutes (240 regulation + 25 per OT).
3. `features.py` — produces `team_features.parquet`, `player_features.parquet`, `elo_ratings.parquet` in `processed/`. Exposes `compute_features_from_data()` for in-memory use.
4. `train.py` — for each date `d`, trains on `game_date < d`, predicts games on `d`. Skips dates below `min_train_games` (cold-start). Saves model + rolling predictions to `outputs/models/`.
5. `evaluate.py` — prints log-loss, Brier score, accuracy; writes plots to `outputs/figures/logreg/` or `outputs/figures/xgboost/`. Elo plots go directly to `outputs/figures/`.

**I/O:** Always use `src/utils/io.py` helpers (`read_raw`, `write_interim`, `read_processed`, etc.) — never call `pd.read_parquet` directly. `SEASON = "2025"` in `io.py` controls all paths. `season_api()` converts `"2025"` → `"2024-25"`.

**No data leakage:** All rolling features use `shift(1).expanding().mean()` per group. Encoding is baked into the parquet so `train.py` can safely slice by `game_date < d`.

**Feature lists (`train.py`):**
- `MODEL_FEATURES` (logreg): `["elo_delta", "home_adv", "fatigue_delta", "acwr_delta", "h2h_delta"]`
- `XGBOOST_MODEL_FEATURES`: full 50+ delta set. Use `tests/feature_tests.py` before expanding features.

**Elo:** Only the home team's row carries `home_adv`. `_load_prior_season_elo()` loads prior-season ratings from `data/<SEASON-1>/processed/` and regresses 50% toward 1500.

**Fatigue:** `fatigue_decay` = exponential decay (λ=0.2/day); `acwr` = 7-day / (28-day/4) minutes. Aggregated to team level as minutes-weighted averages (`season_avg_minutes_decimal`, shift-by-1).

**Column naming:** `_snake()` in `process.py` normalises `UPPER_CASE` and `camelCase` API columns to `snake_case`.

**Playoffs:** Files carry a `_playoffs` suffix. Regular-season data is always included in playoffs runs so Elo and rolling averages aren't cold-started.

## Tests

```bash
python -m pytest tests/    # no disk I/O — DataFrames built in-memory
```

- `test_features.py`, `test_process.py`, `test_train.py` — pytest smoke tests
- `tests/feature_tests.py` — L1 sweep, permutation importance, ablation, VIF utilities (also discovered by pytest)
- `tests/model_tests.py` — standalone script; loads saved model and runs all four analyses

## Tunable Constants

| Constant | File | Default |
|---|---|---|
| `SEASON` | `src/utils/io.py` | `"2025"` |
| `FATIGUE_LAMBDA` | `src/data/features.py` | `0.2` |
| `ELO_K` / `ELO_HOME_ADV_BASE` / `ELO_HOME_ADV_SCALE` / `ELO_HOME_ADV_MIN` | `features.py` | `25` / `40` / `10` / `20` |
| `ELO_MOV_BASELINE` / `ELO_OT_DISCOUNT` | `features.py` | `8.0` / `0.2` |
| `min_train_games` | `src/models/train.py` | `50` |
| `TRAIN_FRACTION` | `src/models/xgboost_grid_search.py` | `0.65` |
| `API_DELAY` / `BULK_API_DELAY` / `COOLDOWN_SECONDS` | `fetch_games.py` | `0.6s` / `1.0s` / `30s` |

XGBoost defaults (`train.py`): `n_estimators=100`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=5`, `gamma=0.0`, `reg_alpha=0.1`, `reg_lambda=0.75`

## Key Notes

- `data/` and `outputs/` are gitignored — all data lives locally only.
- Target milestone: 2026 NBA playoffs.
