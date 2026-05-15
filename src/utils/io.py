"""
io.py — Shared I/O helpers for the NBA stats pipeline.

All functions resolve paths relative to the project root so callers
never need to hard-code directory locations.
"""

import logging
import os
from datetime import date as _date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Project root: src/utils/io.py  →  ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Season subdirectory (e.g. "2026" for the 2025-26 season).
SEASON = os.environ.get("NBA_SEASON", "2026")

RAW_DIR = PROJECT_ROOT / "data" / SEASON / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / SEASON / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / SEASON / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / SEASON
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def season_api(season: str = SEASON) -> str:
    """Convert a season directory label to the nba_api format.

    The project stores data under ``data/<SEASON>/`` where ``SEASON`` is the
    *ending* year (e.g. ``"2026"`` for the 2025-26 season).  The nba_api
    endpoints expect the hyphenated form (``"2025-26"``).

    Examples
    --------
    >>> season_api("2025")
    '2024-25'
    >>> season_api("2026")
    '2025-26'
    """
    year = int(season)
    return f"{year - 1}-{str(year)[2:]}"


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file and return a DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the Parquet file.
    """
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, creating parent directories as needed.

    Parameters
    ----------
    df:
        DataFrame to serialise.
    path:
        Destination path (including filename and .parquet extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def read_raw(filename: str) -> pd.DataFrame:
    """Read a file from ``data/raw/`` by name.

    Parameters
    ----------
    filename:
        E.g. ``"team_gamelog_raw.parquet"``.
    """
    return read_parquet(RAW_DIR / filename)


def write_raw(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to ``data/raw/`` and return the resolved path.

    Parameters
    ----------
    df:
        DataFrame to serialise.
    filename:
        E.g. ``"schedule.parquet"``.
    """
    dest = RAW_DIR / filename
    write_parquet(df, dest)
    return dest


def read_schedule() -> pd.DataFrame:
    """Read the season schedule from ``data/raw/schedule.parquet``."""
    return read_raw("schedule.parquet")


def read_interim(filename: str) -> pd.DataFrame:
    """Read a file from ``data/interim/`` by name.

    Parameters
    ----------
    filename:
        E.g. ``"game_log.parquet"``.
    """
    return read_parquet(INTERIM_DIR / filename)


def write_interim(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to ``data/interim/`` and return the resolved path.

    Parameters
    ----------
    df:
        DataFrame to serialise.
    filename:
        E.g. ``"game_log.parquet"``.
    """
    dest = INTERIM_DIR / filename
    write_parquet(df, dest)
    return dest


def read_processed(filename: str) -> pd.DataFrame:
    """Read a file from ``data/processed/`` by name.

    Parameters
    ----------
    filename:
        E.g. ``"team_features.parquet"``.
    """
    return read_parquet(PROCESSED_DIR / filename)


def write_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to ``data/processed/`` and return the resolved path.

    Parameters
    ----------
    df:
        DataFrame to serialise.
    filename:
        E.g. ``"team_features.parquet"``.
    """
    dest = PROCESSED_DIR / filename
    write_parquet(df, dest)
    return dest


def configure_logging(name: str) -> logging.Logger:
    """Return a named logger with a stdout handler and a daily append-mode file handler.

    Creates ``<project_root>/logs/`` if it does not exist.  Sets
    ``propagate = False`` so the root logger does not duplicate output.
    Safe to call multiple times — duplicate handlers are never added.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(name)-14s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(
        logs_dir / f"pipeline_{_date.today():%Y%m%d}.log",
        mode="a",
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
