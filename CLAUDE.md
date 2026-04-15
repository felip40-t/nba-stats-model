# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Constraints

- Do NOT delete any files, ever. If removal is needed, ask the user to do it manually.
- Do NOT read or edit files outside the project root directory. All file operations must stay within the project folder.

## Environment Setup

```bash
# Activate the virtual environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Common Commands

```bash
# Run a data collection script
python src/data/fetch_games.py

# Launch JupyterLab
jupyter lab

# Run tests (once tests are added)
python -m pytest tests/
```

## Architecture

The project follows a staged data pipeline:

1. **Ingestion** (`src/data/`) — Scripts using `nba_api` fetch raw game/player stats and write to `data/raw/` as Parquet files.
2. **Processing** (`src/data/`) — Cleaning, transformation, and feature engineering (rolling averages, contextual features) produce structured tables in `data/interim/` and `data/processed/`.
3. **Analysis** (`notebooks/`) — Exploratory analysis of team/player trends.
4. **Modelling** (`src/models/`) — Interpretable ML models (logistic regression, linear regression, tree-based) targeting win probability, team points, and player scoring thresholds.
5. **Outputs** (`outputs/`) — Figures and reports generated from models and analysis.

## Key Notes

- `data/` and `outputs/` are gitignored — all data lives locally only.
- Data is stored as Apache Parquet (via `pyarrow`) for efficient I/O.
- The primary data source is `nba_api`; be mindful of rate limits when calling NBA Stats endpoints (add delays between requests if fetching in bulk).
- Target milestone: working software ready for the 2026 NBA playoffs.
