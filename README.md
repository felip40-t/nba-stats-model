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

The project progresses through several stages. The main aim is to have a working software in time for the 2026 NBA playoffs.

## 1. Data Ingestion

Retrieve raw NBA statistics using `nba_api`.

Examples of collected data include:

- Team game logs
- Player game logs
- Box score statistics
- Advanced team metrics

Raw datasets are stored locally and versioned through the project pipeline.

---

## 2. Data Processing

Clean and transform raw datasets into structured tables suitable for analysis.

Examples:

- Game-level datasets
- Team game statistics
- Player game statistics

Feature engineering will include rolling averages and contextual game information.

---

## 3. Exploratory Analysis

Analyse statistical patterns such as:

- Team offensive and defensive performance trends
- Home vs away performance
- Back-to-back scheduling effects
- Player usage and efficiency trends

---

## 4. Predictive Modelling

Build baseline predictive models such as:

- Team win probability
- Team points scored
- Player scoring thresholds

Models will initially focus on interpretable methods such as:

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
│   ├─ raw/
│   ├─ interim/
│   └─ processed/
│
├─ notebooks/
│
├─ src/
│   ├─ data/
│   │   ├─ fetch_games.py
|   |   ├─ process.py
|   |   ├─ features.py
│   │
│   ├─ models/
|   |   ├─ train.py
|   |   ├─ evaluate.py
│   │
│   └─ utils/
|       ├─ io.py
|
│
├─ tests/
│
└─ outputs/
    ├─ figures/
    └─ reports/
```
# Installation
Clone the repository:
```bash
git clone https://github.com/felip40-t/nba-stats-model.git
cd nba-stats-model
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Data collection scripts are located in:

```text
src/data/
```

Example usage:

```bash
python src/data/fetch_games.py
```

This script downloads NBA game data and stores it in the `data/raw` directory.

---

# Data Sources

Primary data source:

- NBA Stats API (via `nba_api`)

Official endpoint documentation:

- https://github.com/swar/nba_api

---

# License

This project is for educational and research purposes.