"""
Shared utility functions — Building Energy Consumption Prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_EXTERNAL = ROOT / "data" / "external"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"
OUTPUTS_MODELS = ROOT / "outputs" / "models"

for _p in [DATA_RAW, DATA_CLEAN, DATA_EXTERNAL, OUTPUTS_FIGURES, OUTPUTS_TABLES, OUTPUTS_MODELS]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)

# ── ASHRAE climate zone lookup (site_id → zone label) ──────────────────────────
# Site coordinates from ASHRAE GPED III metadata
SITE_CLIMATE_ZONE = {
    0: "Hot & Dry",        # Site 0 — Orlando FL area
    1: "Hot & Humid",      # Site 1 — Minneapolis MN area
    2: "Hot & Dry",        # Site 2 — Phoenix AZ area
    3: "Mixed & Humid",    # Site 3 — Charlotte NC area
    4: "Cold",             # Site 4 — London UK area
    5: "Hot & Humid",      # Site 5 — Hong Kong area
    6: "Mixed & Dry",      # Site 6 — Denver CO area
    7: "Cold",             # Site 7 — Ann Arbor MI area
    8: "Hot & Dry",        # Site 8 — Los Angeles CA area
    9: "Hot & Humid",      # Site 9 — Rochester NY area
    10: "Mixed & Humid",   # Site 10 — Philadelphia PA area
    11: "Hot & Dry",       # Site 11 — Austin TX area
    12: "Mixed & Humid",   # Site 12 — Baltimore MD area
    13: "Cold",            # Site 13 — Chicago IL area
    14: "Hot & Humid",     # Site 14 — Washington DC area
    15: "Mixed & Dry",     # Site 15 — Las Vegas NV area
}

# ── Evaluation metrics ─────────────────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Variation of RMSE — ASHRAE Guideline 14 standard metric."""
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        return np.nan
    return float(rmse(y_true, y_pred) / mean_actual * 100)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "CV_RMSE_%": cvrmse(y_true, y_pred),
    }

# ── Meter type label lookup ────────────────────────────────────────────────────
METER_LABELS = {0: "Electricity", 1: "Chilled Water", 2: "Steam", 3: "Hot Water"}

# ── Feature columns used in modelling (single source of truth) ────────────────
FEATURE_COLS = [
    "meter",
    "site_id",
    "log_square_feet",
    "building_age",
    "floor_count",
    "primary_use_enc",
    "use_group_enc",
    "climate_zone_enc",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "air_temperature",
    "dew_temperature",
    "cloud_coverage",
    "wind_speed",
    "sea_level_pressure",
    "precip_depth_1_hr",
    "relative_humidity",
    "HDD",
    "CDD",
    "temp_roll24h",
    "wind_chill",
]

TARGET_COL = "meter_reading"

# ── Building primary use simplified groups ─────────────────────────────────────
USE_GROUP = {
    "Education": "Education",
    "Office": "Office",
    "Entertainment/public assembly": "Public Assembly",
    "Lodging/residential": "Lodging",
    "Healthcare": "Healthcare",
    "Other": "Other",
    "Retail": "Retail",
    "Food sales and service": "Food Service",
    "Warehouse/storage": "Warehouse",
    "Technology/science": "Technology",
    "Manufacturing/industrial": "Industrial",
    "Utility": "Utility",
    "Services": "Services",
    "Parking": "Parking",
    "Religious worship": "Religious",
}
