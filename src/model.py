"""
Model training — Building Energy Consumption Prediction.

Models:
  1. XGBoost           — primary model
  2. LightGBM          — primary comparator
  3. Random Forest     — baseline
  4. Ridge Regression  — linear baseline

Split strategy: temporal holdout — train on Jan–Sep, test on Oct–Dec.
Mirrors realistic deployment where a model trained on historical data
is applied to future time periods.

Usage:
    python src/model.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

from src.utils import (
    DATA_CLEAN, OUTPUTS_MODELS, OUTPUTS_TABLES,
    FEATURE_COLS, TARGET_COL, compute_all_metrics, get_logger,
)

log = get_logger("model")

RANDOM_STATE = 42
# Temporal split: train on Jan–Sep (months 1–9), test on Oct–Dec (months 10–12)
TEST_MONTHS = [10, 11, 12]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_clean_data() -> pd.DataFrame:
    path = DATA_CLEAN / "ashrae_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Clean data not found. Run src/preprocessing.py first."
        )
    log.info("Loading clean data from %s …", path)
    df = pd.read_parquet(path)
    log.info("  Loaded: %d rows × %d cols", *df.shape)
    return df


# ── Temporal train / test split ────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal holdout: train on Jan–Sep, test on Oct–Dec.
    All buildings appear in both sets — mirrors realistic deployment where a
    model trained on historical data is applied to future time periods.
    """
    test_mask = df["month"].isin(TEST_MONTHS)
    train_df = df[~test_mask].copy()
    test_df  = df[test_mask].copy()
    log.info(
        "Temporal split — train: %d rows (Jan–Sep) | test: %d rows (Oct–Dec)",
        len(train_df), len(test_df),
    )
    log.info(
        "  Buildings in train: %d | test: %d | overlap: %d",
        train_df["building_id"].nunique(),
        test_df["building_id"].nunique(),
        len(set(train_df["building_id"]) & set(test_df["building_id"])),
    )
    return train_df, test_df


def get_xy(df: pd.DataFrame, feature_cols: list) -> tuple[np.ndarray, np.ndarray]:
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = np.log1p(df[TARGET_COL].values.astype(np.float32))  # log-transform target
    return X, y, available


# ── Model definitions ──────────────────────────────────────────────────────────

def build_xgboost() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0,
    )


def build_lightgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=10,
        max_features=0.4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_ridge() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])


MODELS = {
    "XGBoost": build_xgboost,
    "LightGBM": build_lightgbm,
    "RandomForest": build_random_forest,
    "Ridge": build_ridge,
}


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(name: str, model, X_train, y_train, X_val=None, y_val=None):
    log.info("Training %s …", name)
    if name == "XGBoost" and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif name == "LightGBM" and X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
    else:
        model.fit(X_train, y_train)
    return model


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test_log) -> dict:
    """Predict in log space; report RMSLE (log-scale) + original-scale MAE/R²/CV(RMSE)."""
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)
    y_pred = np.clip(y_pred, 0, None)
    metrics = compute_all_metrics(y_true, y_pred)
    # RMSLE — primary metric used in ASHRAE GPED-III competition
    metrics["RMSLE"] = float(np.sqrt(
        np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)
    ))
    return metrics


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    log.info("=== Model Training: Building Energy Consumption Prediction ===")

    df = load_clean_data()
    train_df, test_df = temporal_split(df)

    # Use 10% of train as validation for early stopping
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.iloc[:val_size]
    tr_df = train_df.iloc[val_size:]

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    X_tr, y_tr, _ = get_xy(tr_df, feature_cols)
    X_val, y_val, _ = get_xy(val_df, feature_cols)
    X_te, y_te, _ = get_xy(test_df, feature_cols)

    results = {}
    predictions = {}

    for name, builder in MODELS.items():
        model = builder()
        model = train_model(name, model, X_tr, y_tr, X_val, y_val)
        metrics = evaluate_model(model, X_te, y_te)
        results[name] = metrics
        predictions[name] = np.expm1(model.predict(X_te))
        log.info(
            "  %s → RMSLE=%.4f | MAE=%.2f | R²=%.4f | RMSE=%.1f",
            name, metrics.get("RMSLE", float("nan")),
            metrics["MAE"], metrics["R2"], metrics["RMSE"],
        )
        model_path = OUTPUTS_MODELS / f"{name.lower()}_model.joblib"
        joblib.dump(model, model_path)
        log.info("  Saved: %s", model_path)

    # Save results table
    results_df = pd.DataFrame(results).T.round(4)
    results_df.index.name = "Model"
    results_csv = OUTPUTS_TABLES / "model_comparison.csv"
    results_df.to_csv(results_csv)
    log.info("Results table saved: %s", results_csv)
    log.info("\n%s", results_df.to_string())

    # Save test predictions + metadata for evaluation.py
    test_meta = test_df[["building_id", "month", "primary_use_enc", "climate_zone_enc",
                          "use_group_enc", TARGET_COL]].copy()
    test_meta["y_true"] = np.expm1(y_te)
    for name, preds in predictions.items():
        test_meta[f"y_pred_{name}"] = preds
    preds_path = DATA_CLEAN / "test_predictions.parquet"
    test_meta.to_parquet(preds_path, index=False)
    log.info("Predictions saved: %s", preds_path)

    # Save feature column list for evaluation.py
    import json
    (OUTPUTS_MODELS / "feature_cols.json").write_text(json.dumps(feature_cols))

    log.info("=== Training complete ===")


if __name__ == "__main__":
    main()
