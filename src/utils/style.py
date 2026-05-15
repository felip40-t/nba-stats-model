"""Shared dark-theme plotting conventions for all figures in this project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BG    = "#131722"
PANEL = "#1e222d"
TEXT  = "#d1d4dc"
GRID  = "#2a2e39"
BLUE  = "#2962ff"
RED   = "#ff3c00"

# Primary brand colours for all 30 NBA franchises.
NBA_TEAM_COLORS: dict[str, str] = {
    "ATL": "#E03A3E",  # Hawks red
    "BOS": "#007A33",  # Celtics green
    "BKN": "#AAAAAA",  # Nets silver (black → too dark on dark bg)
    "CHA": "#00788C",  # Hornets teal
    "CHI": "#CE1141",  # Bulls red
    "CLE": "#860038",  # Cavaliers wine
    "DAL": "#00538C",  # Mavericks blue
    "DEN": "#FEC524",  # Nuggets gold
    "DET": "#C8102E",  # Pistons red
    "GSW": "#FFC72C",  # Warriors gold
    "HOU": "#CE1141",  # Rockets red
    "IND": "#FDBB30",  # Pacers gold
    "LAC": "#C8102E",  # Clippers red
    "LAL": "#FDB927",  # Lakers gold
    "MEM": "#5D76A9",  # Grizzlies steel blue
    "MIA": "#98002E",  # Heat red
    "MIL": "#00471B",  # Bucks green
    "MIN": "#236192",  # Timberwolves blue
    "NOP": "#85714D",  # Pelicans gold
    "NYK": "#F58426",  # Knicks orange
    "OKC": "#007AC1",  # Thunder blue
    "ORL": "#0077C0",  # Magic blue
    "PHI": "#ED174C",  # Sixers red
    "PHX": "#E56020",  # Suns orange
    "POR": "#E03A3E",  # Blazers red
    "SAC": "#5A2D81",  # Kings purple
    "SAS": "#C4CED4",  # Spurs silver
    "TOR": "#CE1141",  # Raptors red
    "UTA": "#00471B",  # Jazz green
    "WAS": "#E31837",  # Wizards red
}


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=9, length=0)
    ax.grid(which="major", color=GRID, linewidth=0.6, linestyle="-", zorder=1)
    ax.set_axisbelow(True)


def style_fig(fig) -> None:
    fig.patch.set_facecolor(BG)


def legend(ax, **kwargs) -> None:
    ax.legend(
        frameon=True, facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=9,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def styled_subplots(
    figsize: tuple[float, float],
    nrows: int = 1,
    ncols: int = 1,
    dpi: int = 150,
    **kwargs,
) -> tuple:
    """Create a figure/axes pair with the dark theme already applied.

    For multi-panel grids (nrows > 1 or ncols > 1) only the figure background
    is styled here; callers must call style_ax() on each axis individually.
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **kwargs)
    style_fig(fig)
    if nrows == 1 and ncols == 1:
        style_ax(ax)
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(fig, out_path: Path) -> None:
    """Save fig to out_path (creating parent dirs) then close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
