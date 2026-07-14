"""
Shared color and label definitions for PyVWF figures.

All palettes are Okabe–Ito (colorblind-safe, print-friendly).
"""

from __future__ import annotations

# ---------- Base palette (Okabe–Ito) ----------
OKABE_ITO = [
    "#E69F00",  # 0 orange
    "#56B4E9",  # 1 sky blue
    "#009E73",  # 2 bluish green
    "#F0E442",  # 3 yellow
    "#0072B2",  # 4 blue
    "#D55E00",  # 5 vermillion
    "#CC79A7",  # 6 reddish purple
    "#000000",  # 7 black
]

# ---------- Temporal resolution & training ----------
TIME_RES_COLOURS = {
    "fixed": "#D55E00",    # vermillion
    "season": "#009E73",   # bluish green
    "bimonth": "#56B4E9",  # sky blue
    "month": "#E69F00",    # orange
}

TIME_RES_LABELS = {
    "fixed": "Fixed",
    "season": "Seasonal",
    "bimonth": "Bimonthly",
    "month": "Monthly",
}

# Canonical plotting order: coarse -> fine temporal resolution.
TIME_RES_ORDER = {"fixed": 0, "season": 1, "bimonth": 2, "month": 3}

TIME_RES_LINESTYLES = {
    "fixed": "-.",
    "season": ":",
    "bimonth": "--",
    "month": "-",
}

EXISTING_NEW_COLOURS = {
    "Yes": OKABE_ITO[4],   # blue
    "No": OKABE_ITO[5],    # vermillion
}

TURBINE_TYPE_COLOURS = {
    "onshore": OKABE_ITO[4],   # blue
    "offshore": OKABE_ITO[5],  # vermillion
}
