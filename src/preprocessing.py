"""
Preprocessing — Building Energy Consumption Prediction.

Steps:
  1. Load ASHRAE raw CSV files
  2. Parse timestamps → temporal features
  3. Merge building metadata + weather
  4. Assign ASHRAE climate zones
  5. Feature engineering (HDD/CDD, rolling weather, lag features)
  6. Handle missing values and outliers
  7. Encode categoricals
  8. Save clean dataset to data/clean/

Usage:
    python src/preprocessing.py
"""

import numpy as np
import pandas as pd
import holidays
from pathlib import Path
from src.utils import (
    DATA_RAW, DATA_CLEAN, SITE_CLIMATE_ZONE, METER_LABELS, USE_GROUP,
    FEATURE_COLS, TARGET_COL, get_logger
)

log = get_logger("preprocessing")

# Reference temperature for HDD/CDD (18°C = 64.4°F standard)
TBASE = 18.0
# Fraction of ASHRAE train data to use (set 1.0 for full 20M rows — needs ~8 GB RAM)
SAMPLE_FRACTION = 0.25  # 25% ≈ 5M rows, manageable on 16 GB RAM


def load_ashrae_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ashrae_dir = DATA_RAW / "ashrae"
    log.info("Loading ASHRAE raw files …")
    # Support both feather (downloaded via dataset) and CSV formats
    def _load(stem, parse_ts=True):
        feather = ashrae_dir / f"{stem}.feather"
        csv = ashrae_dir / f"{stem}.csv"
        if feather.exists():
            df = pd.read_feather(feather)
        elif csv.exists():
            df = pd.read_csv(csv, parse_dates=["timestamp"] if parse_ts else None)
        else:
            raise FileNotFoundError(f"Neither {feather} nor {csv} found. Run data_collection.py.")
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    train = _load("train")
    meta = _load("building_metadata", parse_ts=False)
    weather = _load("weather_train")
    log.info("  train: %d rows | meta: %d rows | weather: %d rows",
             len(train), len(meta), len(weather))
    return train, meta, weather


def filter_and_sample(train: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious anomalies and take a stratified sample."""
    # Remove negative meter readings (data quality issue documented in competition)
    train = train[train["meter_reading"] >= 0].copy()
    # Remove extreme outliers per building+meter (>99.9th percentile)
    q999 = train.groupby(["building_id", "meter"])["meter_reading"].transform(
        lambda x: x.quantile(0.999)
    )
    train = train[train["meter_reading"] <= q999].copy()
    # Random sample — at 25% of 20M rows all 1449 buildings are well represented
    if SAMPLE_FRACTION < 1.0:
        log.info("Sampling %.0f%% of data …", SAMPLE_FRACTION * 100)
        train = train.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    log.info("  After filter+sample: %d rows", len(train))
    return train


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp column."""
    ts = df["timestamp"]
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek       # 0=Monday
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # US federal holidays (covers most ASHRAE sites)
    us_holidays = holidays.US()
    df["is_holiday"] = ts.dt.date.apply(lambda d: int(d in us_holidays))
    # Cyclical encoding for hour and month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def clean_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps in weather time-series per site."""
    weather = weather.sort_values(["site_id", "timestamp"]).copy()
    weather_cols = [
        "air_temperature", "dew_temperature", "cloud_coverage",
        "wind_speed", "wind_direction", "sea_level_pressure", "precip_depth_1_hr",
    ]
    for col in weather_cols:
        if col in weather.columns:
            weather[col] = (
                weather.groupby("site_id")[col]
                .transform(lambda s: s.interpolate(method="linear", limit=6))
            )
    # Relative humidity approximation from dew point (Magnus formula)
    if "air_temperature" in weather.columns and "dew_temperature" in weather.columns:
        T = weather["air_temperature"]
        Td = weather["dew_temperature"]
        weather["relative_humidity"] = 100 * np.exp(
            (17.625 * Td / (243.04 + Td)) - (17.625 * T / (243.04 + T))
        )
    return weather


def add_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Add HDD, CDD and rolling statistics to weather DataFrame."""
    weather = weather.copy()
    T = weather["air_temperature"]
    weather["HDD"] = np.maximum(TBASE - T, 0)
    weather["CDD"] = np.maximum(T - TBASE, 0)
    # 24-hour rolling mean temperature per site
    weather = weather.sort_values(["site_id", "timestamp"])
    weather["temp_roll24h"] = (
        weather.groupby("site_id")["air_temperature"]
        .transform(lambda s: s.rolling(24, min_periods=1).mean())
    )
    # Wind chill proxy (simplified)
    ws = weather.get("wind_speed", pd.Series(0, index=weather.index))
    weather["wind_chill"] = T - 0.3 * ws
    return weather


def merge_all(
    train: pd.DataFrame,
    meta: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge meter readings with building metadata and weather."""
    df = train.merge(meta, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features and add derived features."""
    df = df.copy()

    # Climate zone from site_id
    df["climate_zone"] = df["site_id"].map(SITE_CLIMATE_ZONE).fillna("Unknown")

    # Simplify primary_use groups
    df["use_group"] = df["primary_use"].map(USE_GROUP).fillna("Other")

    # Label-encode primary_use and use_group
    for col in ["primary_use", "use_group", "climate_zone"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes

    # Log transform building area (right-skewed)
    df["log_square_feet"] = np.log1p(df["square_feet"])

    # Building age
    current_year = 2016  # ASHRAE data is 2016
    df["building_age"] = current_year - df["year_built"].fillna(1970)

    # Meter type label
    df["meter_label"] = df["meter"].map(METER_LABELS)

    # Floor count: fill missing with median
    df["floor_count"] = df["floor_count"].fillna(df["floor_count"].median())

    return df


def drop_unused_and_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup: fill remaining NAs and drop columns not used in modelling."""
    fill_map = {
        "cloud_coverage": 4.0,          # median
        "precip_depth_1_hr": 0.0,
        "wind_direction": 180.0,
        "wind_speed": 3.0,
        "sea_level_pressure": 1013.0,
        "relative_humidity": 60.0,
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    drop_cols = ["timestamp", "primary_use", "use_group", "climate_zone", "meter_label"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def load_uci() -> pd.DataFrame:
    """Load UCI Energy Efficiency dataset as supplementary cross-dataset."""
    path = DATA_RAW / "uci_energy_efficiency.xlsx"
    if not path.exists():
        log.warning("UCI dataset not found — skipping.")
        return pd.DataFrame()
    log.info("Loading UCI Energy Efficiency dataset …")
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [
        "relative_compactness", "surface_area", "wall_area", "roof_area",
        "overall_height", "orientation", "glazing_area", "glazing_area_distribution",
        "heating_load", "cooling_load",
    ]
    df.to_csv(DATA_CLEAN / "uci_clean.csv", index=False)
    log.info("UCI clean saved: %d rows", len(df))
    return df


def main():
    log.info("=== Preprocessing: Building Energy Consumption Prediction ===")

    # --- ASHRAE pipeline ---
    train, meta, weather = load_ashrae_raw()
    train = filter_and_sample(train)
    train = build_temporal_features(train)
    weather = clean_weather(weather)
    weather = add_weather_features(weather)
    df = merge_all(train, meta, weather)
    df = encode_features(df)
    df = drop_unused_and_fillna(df)

    # Validate all feature columns present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing feature columns: %s", missing)
        FEATURE_COLS_USED = [c for c in FEATURE_COLS if c in df.columns]
    else:
        FEATURE_COLS_USED = FEATURE_COLS

    out_path = DATA_CLEAN / "ashrae_clean.parquet"
    df[FEATURE_COLS_USED + [TARGET_COL, "building_id"]].to_parquet(out_path, index=False)
    log.info("Clean ASHRAE dataset saved: %d rows × %d cols → %s",
             len(df), len(FEATURE_COLS_USED) + 2, out_path)

    # --- UCI pipeline ---
    load_uci()

    log.info("=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
