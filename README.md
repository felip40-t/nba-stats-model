# NBA Statistical Modelling Project

> This project was constructed with the assistance of [Claude Code](https://claude.ai/code) by Anthropic.

# Overview

This project builds a reproducible data pipeline and modelling framework for analysing NBA game and player statistics using Python.

The objective is to develop and demonstrate skills in:

- Data acquisition from APIs
- Data cleaning and feature engineering with pandas
- Exploratory statistical analysis
- Machine learning modelling
- Reproducible project structure

The project uses the `nba_api` library to collect official NBA statistics and constructs datasets that can be used to analyse trends and build predictive models.

This repository focuses on **building a clear and extensible workflow**, rather than producing a single black-box prediction model.

---

# Project Goals

The project progresses through several stages. The main aim is to have working software ready for the **2026 NBA playoffs**.

## 1. Data Ingestion

Retrieve raw NBA statistics using `nba_api`.

Data collected per season includes:

- Team game logs (`LeagueGameLog`) — one row per team per game
- Box scores (`BoxScoreTraditionalV3`) — player and team splits per game
- Advanced team metrics derived from box-score totals

Raw datasets are stored locally as Apache Parquet files under `data/<season>/raw/`.

---

## 2. Data Processing

Clean and transform raw datasets into structured interim tables.

Current interim tables (written to `data/<season>/interim/`):

| Table | Description |
|---|---|
| `game_log.parquet` | One row per team per game with contextual flags (`is_home`, `win`, `is_back_to_back`) |
| `player_game_log.parquet` | One row per player per game with box-score stats and decimal minutes |
| `team_advanced.parquet` | Aggregated team totals per game with efficiency metrics (TS%, 3P rate, FT rate, OREB%) |

---

## 3. Feature Engineering

Transform interim tables into model-ready features (written to `data/<season>/processed/`).

### Team features (`team_features.parquet`)

Expanding-window season averages computed from games *prior to* the current one (no data leakage):

- Box-score rolling averages: points, rebounds, assists, steals, blocks, turnovers, FG%, 3P%, FT%, plus-minus
- Efficiency rolling averages: true shooting %, three-point rate, free-throw rate, OREB%
- `season_win_rate` — win percentage entering the game
- `games_played` — number of games played before this one

### Player features (`player_features.parquet`)

Rolling season averages (same expanding, shift-by-1 approach) for points, rebounds, assists, minutes, and shooting splits, plus two fatigue metrics:

**`fatigue_decay`** — Exponential decay load model:

```
fatigue_i = Σ_{j < i}  minutes_j · e^{-λ · (date_i − date_j)}
```

Where `λ = 0.2 day⁻¹` (configurable via `FATIGUE_LAMBDA` in `features.py`). Recent high-minute games contribute the most; load from distant games fades exponentially. A player on a back-to-back carries nearly the full weight of their previous game; after a 5-day break that game contributes only ~37% of its original load.

**`acwr`** — Acute:Chronic Workload Ratio:

```
acwr = (7-day rolling minutes) / (28-day rolling minutes / 4)
```

Both windows exclude the current game. Values above 1.0 signal an acute workload spike above the player's chronic baseline, which sports science literature associates with elevated injury risk.

---

## 4. Exploratory Analysis

Planned analysis of statistical patterns including:

- Team offensive and defensive performance trends
- Home vs away performance
- Back-to-back scheduling effects
- Player usage, fatigue, and efficiency trends

---

## 5. Predictive Modelling

Planned baseline predictive models:

- Team win probability
- Team points scored
- Player scoring thresholds

Models will initially focus on interpretable methods:

- Logistic regression
- Linear regression
- Tree-based models

---

# Technologies Used

### Data & Analysis

| Library | Description |
|---|---|
| [pandas](https://pandas.pydata.org/) | Data manipulation and analysis |
| [numpy](https://numpy.org/) | Numerical computing |
| [scikit-learn](https://scikit-learn.org/) | Machine learning models and utilities |
| [matplotlib](https://matplotlib.org/) | Data visualisation |
| [nba_api](https://github.com/swar/nba_api) | NBA Stats API client |
| [pyarrow](https://arrow.apache.org/docs/python/) | Apache Parquet read/write (efficient data storage) |

### Development Environment

| Tool | Description |
|---|---|
| [Python](https://www.python.org/) | Primary programming language |
| [VS Code](https://code.visualstudio.com/) | Code editor |
| [Git](https://git-scm.com/) | Version control |
| [GitHub](https://github.com/) | Remote repository hosting |
| [JupyterLab](https://jupyterlab.readthedocs.io/) | Interactive notebook environment |

---

# Project Structure

```text
nba-statistics-model/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│   └─ 2025/                    ← season subdirectory (2024-25 season)
│       ├─ raw/                  ← fetched from nba_api, never edited
│       ├─ interim/              ← cleaned & restructured by process.py
│       └─ processed/            ← model-ready features from features.py
│
├─ notebooks/
│
├─ src/
│   ├─ data/
│   │   ├─ fetch_games.py        ← Stage 1: ingest raw data from nba_api
│   │   ├─ process.py            ← Stage 2: clean raw → interim tables
│   │   └─ features.py           ← Stage 3: feature engineering → processed tables
│   │
│   ├─ models/
│   │   ├─ train.py
│   │   └─ evaluate.py
│   │
│   └─ utils/
│       └─ io.py                 ← shared path constants and Parquet helpers
│
├─ tests/
│
└─ outputs/
    ├─ figures/
    └─ reports/
```

The active season is controlled by the `SEASON` constant in `src/utils/io.py`. All path helpers (`read_raw`, `read_interim`, `write_processed`, etc.) resolve beneath `data/<SEASON>/` automatically.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/felip40-t/nba-stats-model.git
cd nba-stats-model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

All scripts are run from the project root. Run them in order:

```bash
# Stage 1 — fetch raw data from nba_api (~90s for 10 games/team)
python src/data/fetch_games.py

# Stage 2 — clean raw data into interim tables
python src/data/process.py

# Stage 3 — feature engineering into processed tables
python src/data/features.py
```

Output is written to `data/2025/{raw,interim,processed}/` respectively.

---

# Data Sources

Primary data source:

- NBA Stats API (via `nba_api`)

Official endpoint documentation:

- https://github.com/swar/nba_api

---

# License

This project is for educational and research purposes.
