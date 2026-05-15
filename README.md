# NBA Statistical Modelling Project

> This project was constructed with the assistance of [Claude Code](https://claude.ai/code) by Anthropic.

## Overview

This project builds a reproducible data pipeline and modelling framework for predicting NBA game outcomes using Python. The objective is to simulate how a win-probability model would perform in a real deployment setting — predicting each game using only information available before it was played, retraining day by day as the season unfolds.

The project uses the `nba_api` library to collect official NBA statistics, engineers a rich set of team and player features, and evaluates models on rolling out-of-sample predictions across the full season.

The target milestone is **working software ready for the 2026 NBA playoffs**.

---

## Current Status

**Version 2.4.0** — Infrastructure and Elo improvements: centralized structured logging to daily files in `logs/`; dark-theme plotting extracted to `src/utils/style.py` with full NBA team colors; outputs are now season-scoped (`outputs/{SEASON}/`); `SEASON` can be overridden via `NBA_SEASON` env var without editing code; neutral-site games (IST finals, Mexico City) processed with `home_adv=0` instead of skipped; `evaluate.py` gains `--output-json` flag; `elo_all_teams.png` now plots the top-10 teams by final rating; Elo hyperparameters retuned. See [Roadmap](#roadmap) for what is still to be done.

| Stage | Status |
|---|---|
| Raw data ingestion (schedule + game logs + box scores) | Done |
| Data processing | Done |
| Feature engineering (team + player) | Done |
| Elo ratings (team-specific home advantage, MOV-adjusted) | Done |
| Player fatigue metrics (decay + ACWR) | Done |
| Efficiency metrics (TS%, 3P rate, FT rate, true OREB%) | Done |
| Defensive metrics (opp pts/OREB/blocks/steals allowed) | Done |
| Recent form windows (5 / 10 / 15 games) | Done |
| Win streak and head-to-head win rate | Done |
| Last-10-game rolling averages (23 stats) | Done |
| Prior-season Elo carry-over (50% regression toward 1500) | Done |
| Elo hyperparameter grid search (`elo_grid_search.py`) | Done |
| Rolling day-by-day training simulation | Done |
| Playoffs model (full regular-season train) | Done (pending train + eval) |
| Model evaluation (metrics + 9 diagnostic plots) | Done |
| Feature importance + collinearity analysis tools | Done |
| Test suite (pytest smoke tests for process, features, train) | Done |
| XGBoost model (alongside logistic regression) | Done |
| XGBoost hyperparameter tuning (`xgboost_grid_search.py`) | Done |
| XGBoost full feature set (`XGBOOST_MODEL_FEATURES`, 50+ deltas) | Done |
| Neutral-site Elo handling (IST finals, Mexico City games) | Done |
| Centralized structured logging → `logs/pipeline_<date>.log` | Done |
| Shared dark-theme plotting utilities (`src/utils/style.py`, NBA team colors) | Done |
| Season-scoped output paths (`outputs/{SEASON}/`) + `NBA_SEASON` env var | Done |
| `evaluate.py --output-json` (writes `latest_metrics.json`) | Done |

---

## Pipeline Architecture

Data flows through five sequential scripts. All are run from the project root.

```
fetch_games.py  →  process.py  →  features.py  →  train.py  →  evaluate.py
   Stage 1          Stage 2        Stage 3         Stage 4       Stage 5
```

### Stage 1 — Ingestion (`src/data/fetch_games.py`)

Fetches the season schedule, team game-logs, and player/team box scores from `nba_api`. Writes three raw Parquet files to `data/<SEASON>/raw/`:

| File | Description |
|---|---|
| `schedule.parquet` | One row per game (including unplayed future games) with home/away team IDs |
| `team_gamelog_raw.parquet` | All team-game rows for the season |
| `boxscore_raw.parquet` | Combined player + team box-score rows tagged with `stat_type` |

**Modes:**
- Default full run: fetches schedule + all game box scores (~20 min for a full season, rate-limited)
- `--schedule-only`: fetch only the schedule, skip box scores
- `--update`: incremental — only fetch game IDs not already on disk (use daily to pick up new results)
- `--playoffs`: same behaviour but queries the Playoffs season type; writes `_playoffs` files
- `--api-delay SEC`: override the per-call delay (default 0.6 s; auto-raised to 1.0 s for runs > 100 new games)

For playoffs, the gamelog and boxscore files carry a `_playoffs` suffix (e.g. `boxscore_raw_playoffs.parquet`).

### Stage 2 — Processing (`src/data/process.py`)

Cleans and restructures raw files into three interim tables in `data/<SEASON>/interim/`:

| Table | Description |
|---|---|
| `game_log.parquet` | One row per team per game with `is_home`, `win`, `is_back_to_back`, `num_ot` flags |
| `player_game_log.parquet` | One row per player per game with box-score stats and decimal minutes |
| `team_advanced.parquet` | Per-game team totals with efficiency metrics (TS%, 3P rate, FT rate, OREB%) |

`num_ot` is derived from summed player-minutes: regulation = 240 per team, each OT adds 25. Pass `--playoffs` to process the `_playoffs` raw files and write `_playoffs` interim files.

### Stage 3 — Feature Engineering (`src/data/features.py`)

Reads interim tables and produces model-ready features in `data/<SEASON>/processed/`. All rolling statistics use `shift(1).expanding().mean()` per group — the current game is never included in its own averages.

**`team_features.parquet`** — one row per team per game:

- Expanding season averages: points, rebounds (total + offensive), assists, steals, blocks, turnovers, personal fouls, FTA, FG%, 3P%, FT%, plus-minus, TS%, 3P rate, FT rate, OREB%
- `season_win_rate`, `games_played` (both entering the current game)
- Elo ratings: `elo_pre`, `elo_post`, `opp_elo_pre`, `home_adv`
- Team-level fatigue aggregates: `team_fatigue`, `team_acwr`

**`player_features.parquet`** — one row per player per game:

- Expanding season averages for all box-score stats
- `days_rest`, `fatigue_decay`, `acwr`

**`elo_ratings.parquet`** — one row per team per game for the full season (~82 rows per team), containing `game_id`, `game_date`, `team_id`, `elo_pre`, `elo_post`. Used by `evaluate.py` to plot how each team's Elo evolves over the season.

Pass `--playoffs` to load `*_playoffs.parquet` interim files alongside the regular-season tables. Regular-season data is always included so Elo ratings carry over and rolling averages are not cold-started. Outputs are filtered to playoff game rows and written as `*_playoffs.parquet` processed files.

When prior-season data exists in `data/<SEASON-1>/processed/`, the pipeline automatically loads each team's final Elo rating and regresses it 50% toward 1500 before the new season begins (`_load_prior_season_elo(carryover=0.5)`). This eliminates the early-season cold-start where all teams are equal. Teams absent from the prior-season snapshot fall back to the flat initial rating.

#### Elo Ratings

Games are replayed chronologically using a margin-of-victory-adjusted Elo formula:

```
expected_home = 1 / (1 + 10 ^ ((R_away − (R_home + H)) / 400))
mov_mult      = log1p(|margin|) / log1p(MOV_BASELINE)
ot_factor     = 1 / (1 + num_ot × OT_DISCOUNT)
R'            = R + K · mov_mult · ot_factor · (S − expected)
```

`mov_mult` scales the update with the final point margin on a log scale — a win by `MOV_BASELINE` points (~8, the median NBA win margin) yields exactly 1.0×; a 30-point blowout ~1.56×; a 1-point win ~0.32×. `ot_factor` discounts overtime wins (1 OT → 0.67×, 2 OT → 0.50×).

The home-court bonus `H` is **team-specific**, scaling with the home team's prior home win rate:

```
H = max(HOME_ADV_MIN, HOME_ADV_BASE + (home_win_rate − 0.5) × HOME_ADV_SCALE)
```

A team with a strong home record earns a larger bonus; the worst home teams are floored at `HOME_ADV_MIN`. Neutral-site games (identified by both rows carrying `is_home=False`) receive `home_adv=0` and do not count toward either team's home record. Current defaults: `HOME_ADV_BASE = 40`, `HOME_ADV_SCALE = 5`, `HOME_ADV_MIN = 15` — tuned via `elo_grid_search.py`.

#### Fatigue Metrics (player-level)

**`fatigue_decay`** — exponential decay load model:

```
fatigue_i = Σ_{j < i}  minutes_j · e^{−λ · (date_i − date_j)}
```

Where `λ = FATIGUE_LAMBDA = 0.2 day⁻¹`. A back-to-back carries nearly the full weight of the previous game; after a 5-day break that game contributes ~37%.

**`acwr`** — Acute:Chronic Workload Ratio:

```
acwr = (7-day rolling minutes) / (28-day rolling minutes / 4)
```

Values above 1.0 signal an acute workload spike above the chronic baseline. Both windows exclude the current game. Player metrics are aggregated to the team level (minutes-weighted mean) to produce `team_fatigue` and `team_acwr`.

#### In-memory computation

`features.py` exposes `compute_features_from_data(game_log, player_game_log, team_advanced, cutoff_date=None)` for use by the rolling training loop. When `cutoff_date=d` is given, only games on or before `d` are used and nothing is written to disk. Running `features.py` directly (no cutoff) saves the end-of-season snapshot.

### Stage 4 — Training (`src/models/train.py`)

**Rolling simulation (default):** Loads the full-season processed feature snapshot from disk once upfront. Because the feature table uses `shift(1).expanding()`, each game row already carries only pre-game information — no per-iteration recompute is needed. For each unique game date `d`:

1. Train a fresh model on all complete rows with `game_date < d`.
2. Predict every game on date `d`.

Dates with fewer than `min_train_games` (default 50) complete training rows are skipped (early-season cold start). After the loop, a final model is trained on the full season and saved.

**Playoffs mode (`--playoffs`):** Trains a single model on the entire regular-season processed feature snapshot. Use after the regular season ends to prepare a model ready to predict playoff match-ups.

**Model selection (`--model`):** Two models are available and share the same pipeline and feature set:

| Model | Flag | Description |
|---|---|---|
| Logistic regression | `--model logreg` (default) | `StandardScaler + LogisticRegression` sklearn pipeline |
| XGBoost | `--model xgboost` | `XGBClassifier` with hyperparameters tuned via `xgboost_grid_search.py`; best params stored in `XGBOOST_DEFAULT_PARAMS` |

**Feature lists (home − away deltas):**

Two separate feature lists are maintained. Logistic regression uses a short, regularised set; XGBoost uses the full delta set computed by `compute_deltas()`.

`MODEL_FEATURES` (logistic regression — 5 features):

| Feature | Description |
|---|---|
| `elo_delta` | `home_elo_pre − away_elo_pre` |
| `home_adv` | Home team's team-specific Elo home-court bonus |
| `fatigue_delta` | Team fatigue (minutes-weighted exponential decay) |
| `acwr_delta` | Team Acute:Chronic Workload Ratio |
| `h2h_delta` | Head-to-head win rate between these two teams |

`XGBOOST_MODEL_FEATURES` (XGBoost — 50+ features): the full set of home-minus-away deltas computed by `compute_deltas()`, covering season averages, efficiency, rebounding, playmaking, defensive metrics, recent form, win streak, fatigue, head-to-head, and 23 last-10-game rolling stats. Use `tests/model_tests.py` to analyse feature importance and collinearity.

**Outputs** (model name is substituted for `{model}`):

| File | Description |
|---|---|
| `outputs/models/win_probability_{model}.joblib` | Final fitted model (full regular season) |
| `outputs/models/win_probability_{model}_playoffs.joblib` | Model trained on full regular season for playoff use (`--playoffs` mode) |
| `outputs/models/rolling_predictions_{model}.parquet` | Each game predicted once using only prior-date data |

### Stage 5 — Evaluation (`src/models/evaluate.py`)

Loads the rolling predictions and saved model for the selected `--model`, prints log-loss, Brier score, and accuracy, then saves diagnostic plots. Model-specific plots go to `outputs/figures/{model}/`; Elo plots are model-agnostic and stay at `outputs/figures/`.

**Model-specific plots** (`outputs/figures/logreg/` or `outputs/figures/xgboost/`):

| File | Description |
|---|---|
| `calibration_curve.png` | Predicted probability vs. actual win rate (scatter, colour-coded by bucket size) |
| `feature_coefficients.png` | Logistic regression coefficients sorted by magnitude (logreg only) |
| `feature_importance.png` | XGBoost feature importances by gain (xgboost only) |
| `rolling_accuracy.png` | Weekly accuracy bars + rolling-window accuracy line over the season |
| `roc_curve.png` | ROC curve with AUC |
| `confidence_histogram.png` | Distribution of predicted home-win probabilities by actual outcome |
| `accuracy_by_confidence.png` | Accuracy binned by model confidence level (`max(p, 1−p)`) |
| `team_accuracy.png` | Per-team prediction accuracy ranked bar chart |

**Model-agnostic plots** (`outputs/figures/`):

| File | Description |
|---|---|
| `elo_time_series.png` | Per-team Elo rating progression — small multiples (one panel per team) |
| `elo_all_teams.png` | Top-10 teams by final Elo rating on a single chart, each coloured by franchise; all plotted teams labelled at the end of their line |

---

## Roadmap

The items below are planned for future versions, roughly in priority order.

### Elo & home-court

- **Bayesian prior on home-court advantage** — replace the hard-scaled `H` formula with a Beta-posterior updated from observed home results; prevents wild swings from small samples early in the season

### Feature engineering

- **Opponent-adjusted stats** — rolling averages accounting for the quality of opponents faced
- **Home/away splits** — separate rolling averages for home and away performance
- **Opponent FG% allowed** — currently only volume defensive stats are tracked; allowing FG% per game would add shot quality
- **Pace and possession metrics** — possessions per game, offensive and defensive rating
- **Travel fatigue** — miles traveled and time zones crossed before each game; orthogonal to minutes-based fatigue
- **Schedule density** — rolling games-per-7-days; captures cumulative compression beyond individual back-to-backs

### Model inputs

- **Calibration post-processing** — Platt scaling or isotonic regression on rolling predictions

### Player features

- **Player experience** — season number, career games played, age
- **Player form** — short-window deviations from seasonal average for points and minutes
- **Injury and rest context** — extended days-off tracking, returning-from-absence flags
- **Rotation stability** — variance in recent minutes as a proxy for role certainty

---

## Tunable Constants

| Constant | File | Default | Purpose |
|---|---|---|---|
| `SEASON` | `src/utils/io.py` | `"2025"` (Python) / `"2026"` (Makefile) | Active season; controls all data paths. Override via `NBA_SEASON` env var |
| `API_DELAY` | `src/data/fetch_games.py` | `0.6` s | Delay between `nba_api` calls for small runs |
| `BULK_API_DELAY` | `src/data/fetch_games.py` | `1.0` s | Delay used when fetching > 100 new games |
| `CHECKPOINT_EVERY` | `src/data/fetch_games.py` | `100` | Games between intermediate box-score saves |
| `COOLDOWN_EVERY` | `src/data/fetch_games.py` | `200` | Games between long cooldown pauses |
| `COOLDOWN_SECONDS` | `src/data/fetch_games.py` | `30.0` s | Duration of each cooldown pause |
| `FATIGUE_LAMBDA` | `src/data/features.py` | `0.25` | Decay rate for exponential fatigue model (day⁻¹) |
| `ELO_INITIAL` | `src/data/features.py` | `1500.0` | Starting Elo for all teams |
| `ELO_K` | `src/data/features.py` | `25.0` | Elo K-factor (base update step size) |
| `ELO_HOME_ADV_BASE` | `src/data/features.py` | `40.0` | Elo home-court bonus for a .500 home record |
| `ELO_HOME_ADV_SCALE` | `src/data/features.py` | `5.0` | Sensitivity of home-court bonus to home win rate |
| `ELO_HOME_ADV_MIN` | `src/data/features.py` | `15.0` | Floor for worst home-record teams |
| `ELO_MOV_BASELINE` | `src/data/features.py` | `8.0` | Point margin yielding a MOV multiplier of 1.0 (≈ median NBA win margin) |
| `ELO_OT_DISCOUNT` | `src/data/features.py` | `0.1` | K reduction per OT period: `ot_factor = 1 / (1 + num_ot × discount)` |
| `ELO_CARRYOVER` | `src/data/features.py` | `0.55` | Fraction of prior-season Elo deviation from 1500 that carries over |
| `min_train_games` | `src/models/train.py` | `50` | Minimum training rows before rolling predictions begin |
| `XGBOOST_DEFAULT_PARAMS.n_estimators` | `src/models/train.py` | `100` | Number of XGBoost trees |
| `XGBOOST_DEFAULT_PARAMS.max_depth` | `src/models/train.py` | `4` | Maximum tree depth |
| `XGBOOST_DEFAULT_PARAMS.learning_rate` | `src/models/train.py` | `0.03` | Step size shrinkage |
| `XGBOOST_DEFAULT_PARAMS.subsample` | `src/models/train.py` | `0.7` | Row subsampling ratio per tree |
| `XGBOOST_DEFAULT_PARAMS.colsample_bytree` | `src/models/train.py` | `0.7` | Column subsampling ratio per tree |
| `XGBOOST_DEFAULT_PARAMS.min_child_weight` | `src/models/train.py` | `5` | Minimum sum of instance weight in a leaf |
| `XGBOOST_DEFAULT_PARAMS.gamma` | `src/models/train.py` | `0.0` | Minimum loss reduction required to split a node |
| `XGBOOST_DEFAULT_PARAMS.reg_alpha` | `src/models/train.py` | `0.1` | L1 regularisation |
| `XGBOOST_DEFAULT_PARAMS.reg_lambda` | `src/models/train.py` | `0.75` | L2 regularisation |

---

## Project Structure

```text
nba-stats-model/
│
├─ README.md
├─ pyproject.toml                   ← build system, pytest config, ruff config
├─ requirements.txt
├─ Makefile                         ← convenience targets (see Common Commands)
├─ .gitignore
│
├─ data/
│   └─ 2025/                       ← season subdirectory (2024-25 season)
│       ├─ raw/                     ← fetched from nba_api, never edited
│       ├─ interim/                 ← cleaned & restructured by process.py
│       └─ processed/               ← model-ready features from features.py
│
├─ src/
│   ├─ data/
│   │   ├─ fetch_games.py           ← Stage 1: ingest schedule + raw data from nba_api
│   │   ├─ process.py               ← Stage 2: clean raw → interim tables
│   │   └─ features.py              ← Stage 3: feature engineering
│   │
│   ├─ models/
│   │   ├─ train.py                 ← Stage 4: rolling training simulation + playoffs model
│   │   ├─ evaluate.py             ← Stage 5: evaluation and diagnostic plots
│   │   ├─ elo_grid_search.py      ← Grid search over Elo hyperparameters (standalone)
│   │   └─ xgboost_grid_search.py  ← Grid search over XGBoost hyperparameters (standalone)
│   │
│   └─ utils/
│       ├─ io.py                    ← shared path constants, Parquet helpers, configure_logging()
│       ├─ display.py               ← CLI table-printing helper (print_table)
│       └─ style.py                 ← dark-theme plotting constants, NBA team colors, helpers
│
├─ tests/
│   ├─ conftest.py                  ← adds project root to sys.path for pytest
│   ├─ test_process.py              ← smoke tests for process.py
│   ├─ test_features.py             ← smoke tests for features.py
│   ├─ test_train.py                ← smoke tests for train.py
│   ├─ feature_tests.py             ← feature analysis utilities: L1 sweep, permutation importance, ablation, VIF
│   └─ model_tests.py               ← script: run feature_tests against the trained model on disk
│
├─ logs/                            ← daily pipeline log files (pipeline_<YYYYMMDD>.log)
└─ outputs/
    └─ 2026/                        ← season-scoped (controlled by NBA_SEASON env var)
        ├─ models/                  ← fitted models + rolling predictions (named by model type)
        └─ figures/
            ├─ logreg/              ← logreg-specific diagnostic plots
            ├─ xgboost/             ← xgboost-specific diagnostic plots
            └─ (root)               ← model-agnostic Elo plots
```

---

## Installation

```bash
git clone https://github.com/felip40-t/nba-stats-model.git
cd nba-stats-model
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Running the Pipeline

All scripts are run from the project root in order:

```bash
# Stage 1 — fetch schedule + raw data from nba_api (~20 min for full season)
python src/data/fetch_games.py

# Stage 2 — clean raw → interim tables
python src/data/process.py

# Stage 3 — feature engineering → processed tables + elo_ratings.parquet
python src/data/features.py

# Stage 4 — rolling day-by-day training simulation (logistic regression)
python src/models/train.py
# or with XGBoost:
python src/models/train.py --model xgboost

# Stage 5 — evaluate and plot
python src/models/evaluate.py
# or with XGBoost:
python src/models/evaluate.py --model xgboost
```

**Optional — Elo hyperparameter grid search:**

```bash
# Search over K, home_adv_*, mov_baseline, ot_discount, carryover, inv_scale
python src/models/elo_grid_search.py
# Results saved to outputs/models/elo_grid_search_results.parquet
```

**Optional — XGBoost hyperparameter grid search:**

```bash
# Chronological holdout search (65% train / 35% test) over n_estimators, max_depth,
# learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_lambda
python src/models/xgboost_grid_search.py
# Results saved to outputs/models/xgboost_grid_search_results.parquet
```

**Common flags:**

```bash
# Fetch only the schedule (no box scores)
python src/data/fetch_games.py --schedule-only

# Pick up newly-played games without re-fetching the full history
python src/data/fetch_games.py --update

# Override the API call delay (seconds)
python src/data/fetch_games.py --api-delay 1.5

# Full playoffs pipeline (run after completing the regular-season pipeline above)
python src/data/fetch_games.py --playoffs [--update]
python src/data/process.py --playoffs
python src/data/features.py --playoffs

# Train a playoffs-ready model on all regular-season data
python src/models/train.py --playoffs
python src/models/train.py --model xgboost --playoffs

# Raise the cold-start threshold (default 50)
python src/models/train.py --min-train-games 100

# Write evaluation metrics to outputs/{SEASON}/models/latest_metrics.json
python src/models/evaluate.py --output-json
python src/models/evaluate.py --model xgboost --output-json
```

The active season is controlled by `SEASON` in `src/utils/io.py` (default `"2025"`). Override it without editing code via the `NBA_SEASON` env var — e.g. `NBA_SEASON=2026 make pipeline` or `NBA_SEASON=2026 python src/data/fetch_games.py`. Outputs are written to `outputs/{SEASON}/`.

**Run the test suite:**

```bash
python -m pytest tests/
```

---

## Technologies Used

| Library | Purpose |
|---|---|
| [pandas](https://pandas.pydata.org/) | Data manipulation and feature engineering |
| [numpy](https://numpy.org/) | Numerical computing |
| [scikit-learn](https://scikit-learn.org/) | ML models, scaling, metrics |
| [matplotlib](https://matplotlib.org/) | Visualisation |
| [nba_api](https://github.com/swar/nba_api) | NBA Stats API client |
| [pyarrow](https://arrow.apache.org/docs/python/) | Apache Parquet I/O |
| [xgboost](https://xgboost.readthedocs.io/) | Gradient boosting model |
| [joblib](https://joblib.readthedocs.io/) | Model serialisation |

---

## License

This project is for educational and research purposes.
