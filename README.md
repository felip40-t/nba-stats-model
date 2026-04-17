# NBA Statistical Modelling Project

> This project was constructed with the assistance of [Claude Code](https://claude.ai/code) by Anthropic.

## Overview

This project builds a reproducible data pipeline and modelling framework for predicting NBA game outcomes using Python. The objective is to simulate how a win-probability model would perform in a real deployment setting — predicting each game using only information available before it was played, retraining day by day as the season unfolds.

The project uses the `nba_api` library to collect official NBA statistics, engineers a rich set of team and player features, and evaluates models on rolling out-of-sample predictions across the full season.

The target milestone is **working software ready for the 2026 NBA playoffs**.

---

## Current Status

**Version 1.0** — The full pipeline is operational end-to-end on test data (2024-25 season). The rolling training simulation has been designed and coded; further refinements are needed before running it on a full season. See [Roadmap](#roadmap) for what is still to be done.

| Stage | Status |
|---|---|
| Raw data ingestion (schedule + game logs + box scores) | Done |
| Data processing | Done |
| Feature engineering (team + player) | Done |
| Elo ratings (team-specific home advantage) | Done |
| Player fatigue metrics (decay + ACWR) | Done |
| Rolling day-by-day training simulation | Designed, needs optimisation |
| Model evaluation | Done (pending full-season run) |

---

## Pipeline Architecture

Data flows through four sequential scripts. All are run from the project root.

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
- `--playoffs`: same behaviour but queries the Playoffs season type

### Stage 2 — Processing (`src/data/process.py`)

Cleans and restructures raw files into three interim tables in `data/<SEASON>/interim/`:

| Table | Description |
|---|---|
| `game_log.parquet` | One row per team per game with `is_home`, `win`, `is_back_to_back` flags |
| `player_game_log.parquet` | One row per player per game with box-score stats and decimal minutes |
| `team_advanced.parquet` | Per-game team totals with efficiency metrics (TS%, 3P rate, FT rate, OREB%) |

### Stage 3 — Feature Engineering (`src/data/features.py`)

Reads interim tables and produces model-ready features in `data/<SEASON>/processed/`. All rolling statistics use `shift(1).expanding().mean()` per group — the current game is never included in its own averages.

**`team_features.parquet`** — one row per team per game:

- Expanding season averages: points, rebounds, assists, steals, blocks, turnovers, FG%, 3P%, FT%, plus-minus, TS%, 3P rate, FT rate, OREB%
- `season_win_rate`, `games_played` (both entering the current game)
- Elo ratings: `elo_pre`, `elo_post`, `opp_elo_pre`, `home_adv`
- Team-level fatigue aggregates: `team_fatigue`, `team_acwr`

**`player_features.parquet`** — one row per player per game:

- Expanding season averages for all box-score stats
- `days_rest`, `fatigue_decay`, `acwr`

**`elo_ratings.parquet`** — one row per team per game for the full season (~82 rows per team), containing `game_date`, `team_id`, `elo_pre`, `elo_post`. Used to plot how each team's Elo evolves over the season.

#### Elo Ratings

Games are replayed chronologically using the standard Elo formula:

```
expected_home = 1 / (1 + 10 ^ ((R_away − (R_home + H)) / 400))
R'            = R + K · (S − expected)
```

The home-court bonus `H` is **team-specific**, scaling with the home team's prior home win rate:

```
H = max(HOME_ADV_MIN, HOME_ADV_BASE + (home_win_rate − 0.5) × HOME_ADV_SCALE)
```

A team with a strong home record earns a larger bonus; the worst home teams are floored at `HOME_ADV_MIN`.

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

Rather than a static train/test split, training runs a **rolling day-by-day simulation** that mirrors real deployment:

1. For each unique game date `d` in the season:
   - Recompute features in memory for all games up to and including `d`. Because rolling stats use `shift(1)`, game rows on date `d` carry only pre-`d` information — zero leakage.
   - Train a fresh logistic regression on all completed games with `game_date < d`.
   - Predict every game on date `d`.
2. After the loop, train a final model on the full season and save it.

**Model inputs** (home − away deltas):

| Feature | Description |
|---|---|
| `elo_delta` | `home_elo_pre − away_elo_pre` |
| `home_adv` | Home team's team-specific Elo home-court bonus |
| `win_rate_delta` | Season win rate differential |
| `pts_delta` | Season average points differential |
| `fg_pct_delta` | Season FG% differential |
| `fatigue_delta` | Team fatigue differential |
| `acwr_delta` | ACWR differential |

**Outputs:**

| File | Description |
|---|---|
| `outputs/models/win_probability_logreg.joblib` | Final fitted pipeline (full season) |
| `outputs/models/rolling_predictions.parquet` | Each game predicted once using only prior-date data |

### Stage 5 — Evaluation (`src/models/evaluate.py`)

Loads `rolling_predictions.parquet` and prints log-loss, Brier score, and accuracy. Saves a calibration curve and feature-coefficient chart to `outputs/figures/`.

---

## Roadmap

The items below are planned for future versions, roughly in priority order.

### Performance optimisation of the rolling simulation

The current fatigue computation (`_fatigue_decay_player`) is O(n²) per player. For a full 82-game season with ~450 active players, recomputing features from scratch on every game day makes the simulation slow. Planned fixes:

- Incremental Elo updates — carry forward the previous day's ratings rather than replaying from game 1
- Incremental fatigue computation — update rather than recompute from scratch
- Vectorise the inner fatigue loop with NumPy broadcasting

### Additional rolling statistics

- **Opponent-adjusted stats** — rolling averages accounting for the quality of opponents faced
- **Recent form** — short-window (last 5–10 games) rolling averages alongside the season-long expanding average to capture momentum and slumps
- **Home/away splits** — separate rolling averages for home and away performance
- **Defensive metrics** — opponent points allowed, opponent FG% allowed, defensive rating
- **Pace and possession metrics** — possessions per game, offensive and defensive rating

### Advanced player features

- **Plus-minus** — raw `+/-` is already in the box score; rolling `+/-` averages will be incorporated as a team and player feature
- **Player experience** — season number, career games played, age; relevant for reliability and expected variance
- **Player form and momentum** — short-window deviations from seasonal average for points and minutes; captures hot/cold streaks and whether a player is being leaned on more than usual
- **Injury and rest context** — extended days-off tracking, whether a player is returning from a known absence, back-to-back fatigue flags
- **Rotation stability** — variance in minutes across recent games as a proxy for role certainty

### Model improvements

- **Gradient boosting** (XGBoost / LightGBM) as a second baseline alongside logistic regression
- **Calibration post-processing** — Platt scaling or isotonic regression to improve probability reliability
- **Feature selection** — systematic comparison of L1 vs L2 regularisation

### Infrastructure

- Jupyter notebooks for exploratory analysis and Elo time-series visualisation
- Automated full-season pipeline runner with progress logging

---

## Tunable Constants

| Constant | File | Default | Purpose |
|---|---|---|---|
| `SEASON` | `src/utils/io.py` | `"2025"` | Active season; controls all data paths |
| `API_DELAY` | `src/data/fetch_games.py` | `0.6` s | Delay between `nba_api` calls |
| `FATIGUE_LAMBDA` | `src/data/features.py` | `0.2` | Decay rate for exponential fatigue model |
| `ELO_INITIAL` | `src/data/features.py` | `1500.0` | Starting Elo for all teams |
| `ELO_K` | `src/data/features.py` | `20.0` | Elo K-factor (update step size) |
| `ELO_HOME_ADV_BASE` | `src/data/features.py` | `100.0` | Elo home-court bonus for a .500 home record |
| `ELO_HOME_ADV_SCALE` | `src/data/features.py` | `100.0` | Sensitivity to home win rate |
| `ELO_HOME_ADV_MIN` | `src/data/features.py` | `50.0` | Floor for worst home-record teams |

---

## Project Structure

```text
nba-stats-model/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│   └─ 2025/                       ← season subdirectory (2024-25 season)
│       ├─ raw/                     ← fetched from nba_api, never edited
│       ├─ interim/                 ← cleaned & restructured by process.py
│       └─ processed/               ← model-ready features from features.py
│
├─ notebooks/
│
├─ src/
│   ├─ data/
│   │   ├─ fetch_games.py           ← Stage 1: ingest schedule + raw data from nba_api
│   │   ├─ process.py               ← Stage 2: clean raw → interim tables
│   │   └─ features.py              ← Stage 3: feature engineering
│   │
│   ├─ models/
│   │   ├─ train.py                 ← Stage 4: rolling training simulation
│   │   └─ evaluate.py             ← Stage 5: evaluation and plots
│   │
│   └─ utils/
│       └─ io.py                    ← shared path constants and Parquet helpers
│
├─ tests/
│
└─ outputs/
    ├─ models/                      ← fitted model + rolling predictions
    └─ figures/                     ← calibration curve, feature coefficients
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

# Stage 4 — rolling day-by-day training simulation
python src/models/train.py

# Stage 5 — evaluate and plot
python src/models/evaluate.py
```

To fetch only the schedule without box scores:

```bash
python src/data/fetch_games.py --schedule-only
```

To pick up newly-played games without re-fetching the full history:

```bash
python src/data/fetch_games.py --update
```

The active season is controlled by `SEASON` in `src/utils/io.py`. Changing it to a new year automatically reroutes all data paths.

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
| [joblib](https://joblib.readthedocs.io/) | Model serialisation |
| [JupyterLab](https://jupyterlab.readthedocs.io/) | Interactive exploration |

---

## License

This project is for educational and research purposes.
