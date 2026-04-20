"""Shared display helpers for CLI output."""

from __future__ import annotations

import pandas as pd


def print_table(title: str, df: pd.DataFrame, max_cols: int = 12) -> None:
    """Pretty-print a DataFrame to stdout with a heading.

    Shows all rows but limits visible columns to ``max_cols`` per block so the
    output stays readable in a terminal without horizontal scrolling.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(sep)

    cols = list(df.columns)
    for start in range(0, len(cols), max_cols):
        chunk = cols[start : start + max_cols]
        print(df[chunk].to_string(index=True))
        if start + max_cols < len(cols):
            print()
