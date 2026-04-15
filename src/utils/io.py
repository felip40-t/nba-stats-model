"""
io.py — Shared I/O helpers for the NBA stats pipeline.

All functions resolve paths relative to the project root so callers
never need to hard-code directory locations.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Project root: src/utils/io.py  →  ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


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
