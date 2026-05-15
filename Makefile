PYTHON := .venv/bin/python
SEASON ?= 2026
export NBA_SEASON = $(SEASON)

# ── Season selection ──────────────────────────────────────────────────────────
# Pass SEASON=<year> to any target to run the pipeline for a different season.
# The year is the *ending* year of the season (e.g. 2026 = 2025-26 season).
#
#   make pipeline                        # 2025-26 season (default)
#   make pipeline SEASON=2024            # 2023-24 season
#   make pipeline-xgboost SEASON=2024
#   make fetch SEASON=2024
#   make fetch-playoffs SEASON=2024
#   make process SEASON=2024
#   make features SEASON=2024
#   make train SEASON=2024
#   make evaluate SEASON=2024
# ────────────────────────────────────────────────────────────────────────────

.PHONY: all pipeline pipeline-xgboost pipeline-playoffs pipeline-xgboost-playoffs \
        fetch fetch-update fetch-schedule fetch-playoffs fetch-playoffs-update \
        process process-playoffs \
        features features-playoffs \
        train train-xgboost train-playoffs train-xgboost-playoffs \
        evaluate evaluate-xgboost \
        elo-grid-search xgboost-grid-search \
        test lint format

# ── Full pipelines ────────────────────────────────────────────────────────────

all: pipeline

pipeline: fetch process features train evaluate

pipeline-xgboost: fetch process features train-xgboost evaluate-xgboost

pipeline-playoffs: fetch-playoffs process-playoffs features-playoffs train-playoffs

pipeline-xgboost-playoffs: fetch-playoffs process-playoffs features-playoffs train-xgboost-playoffs

# ── Stage 1: fetch ────────────────────────────────────────────────────────────

fetch:
	$(PYTHON) src/data/fetch_games.py

fetch-update:
	$(PYTHON) src/data/fetch_games.py --update

fetch-schedule:
	$(PYTHON) src/data/fetch_games.py --schedule-only

fetch-playoffs:
	$(PYTHON) src/data/fetch_games.py --playoffs

fetch-playoffs-update:
	$(PYTHON) src/data/fetch_games.py --playoffs --update

# ── Stage 2: process ──────────────────────────────────────────────────────────

process:
	$(PYTHON) src/data/process.py

process-playoffs:
	$(PYTHON) src/data/process.py --playoffs

# ── Stage 3: features ─────────────────────────────────────────────────────────

features:
	$(PYTHON) src/data/features.py

features-playoffs:
	$(PYTHON) src/data/features.py --playoffs

# ── Stage 4: train ────────────────────────────────────────────────────────────

train:
	$(PYTHON) src/models/train.py

train-xgboost:
	$(PYTHON) src/models/train.py --model xgboost

train-playoffs:
	$(PYTHON) src/models/train.py --playoffs

train-xgboost-playoffs:
	$(PYTHON) src/models/train.py --model xgboost --playoffs

# ── Stage 5: evaluate ─────────────────────────────────────────────────────────

evaluate:
	$(PYTHON) src/models/evaluate.py

evaluate-xgboost:
	$(PYTHON) src/models/evaluate.py --model xgboost

# ── Optional: grid searches ───────────────────────────────────────────────────

elo-grid-search:
	$(PYTHON) src/models/elo_grid_search.py

xgboost-grid-search:
	$(PYTHON) src/models/xgboost_grid_search.py

# ── Dev tools ─────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src/

format:
	$(PYTHON) -m ruff format src/
