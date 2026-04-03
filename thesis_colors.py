"""
Shared color and label definitions for PyVWF thesis figures (Chapters 3–5).

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

# ---------- Chapter 3: Temporal resolution & training ----------
TIME_RES_COLOURS = {
    "fixed": "#D55E00",    # vermillion
    "season": "#009E73",   # bluish green
    "bimonth": "#56B4E9",  # sky blue
    "month": "#E69F00",    # orange
}

EXISTING_NEW_COLOURS = {
    "Yes": OKABE_ITO[4],   # blue
    "No": OKABE_ITO[5],    # vermillion
}

# ---------- Chapter 4: Grid interpolation ----------
METHOD_COLOURS = {
    "nearest": OKABE_ITO[7],   # black
    "idw": OKABE_ITO[4],       # blue
    "rbf": OKABE_ITO[5],       # vermillion
    "kriging": OKABE_ITO[2],   # bluish green
}

COUNTRY_COLOURS = {
    "DE": OKABE_ITO[4],   # blue
    "DK": OKABE_ITO[0],   # orange
    "UK": OKABE_ITO[2],   # bluish green
    "FR": OKABE_ITO[5],   # vermillion
    "ES": OKABE_ITO[6],   # reddish purple
    "NL": OKABE_ITO[1],   # sky blue
    "BE": OKABE_ITO[3],   # yellow
    "SE": OKABE_ITO[7],   # black
    "NO": "#bcbd22",       # olive (extended)
    "PT": "#17becf",       # teal (extended)
    "IE": "#aec7e8",       # light blue (extended)
    "IT": "#ffbb78",       # light orange (extended)
}

# ---------- Chapter 5: ML models ----------
MODEL_COLOURS = {
    "ridge": OKABE_ITO[4],              # blue
    "gradient_boosting": OKABE_ITO[5],  # vermillion
    "lightgbm": OKABE_ITO[2],           # bluish green
    "xgboost": OKABE_ITO[6],            # reddish purple
    "random_forest": OKABE_ITO[1],      # sky blue
    "lasso": OKABE_ITO[0],              # orange
    "elastic_net": OKABE_ITO[7],        # black
    "mlp": OKABE_ITO[3],                # yellow
}

GROUP_COLOURS = {
    "terrain": OKABE_ITO[1],   # sky blue
    "era5": OKABE_ITO[0],      # orange
    "turbine": OKABE_ITO[2],   # bluish green
    "fleet": OKABE_ITO[2],     # bluish green (same as turbine)
    "corine": OKABE_ITO[6],    # reddish purple
    "spatial": OKABE_ITO[3],   # yellow
}
