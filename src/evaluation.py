"""
Evaluation and figure generation — Building Energy Consumption Prediction.

Generates all 5 publication figures and LaTeX tables.
Primary metrics: RMSLE (ASHRAE competition standard) and R² on log scale.

  Fig 1 — Predicted vs Actual scatter (log scale) by building use group
  Fig 2 — SHAP global beeswarm (XGBoost, top-15 features)
  Fig 3 — SHAP climate-zone mean|SHAP| heatmap
  Fig 4 — Model comparison bar chart (RMSLE, MAE, R²_log)
  Fig 5 — SHAP dependence: air_temperature and log_square_feet

Usage:
    python src/evaluation.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import joblib

from src.utils import (
    DATA_CLEAN, OUTPUTS_FIGURES, OUTPUTS_MODELS, OUTPUTS_TABLES,
    SITE_CLIMATE_ZONE, FEATURE_COLS, TARGET_COL,
    rmse, mae, r2, cvrmse, get_logger,
)

# ── Correct metrics for this dataset ──────────────────────────────────────────
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error — primary ASHRAE competition metric."""
    return float(np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)) ** 2)))

def r2_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² computed in log space, robust to extreme outliers."""
    log_true = np.log1p(y_true)
    log_pred = np.log1p(np.clip(y_pred, 0, None))
    ss_res = np.sum((log_true - log_pred) ** 2)
    ss_tot = np.sum((log_true - log_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_c = np.clip(y_pred, 0, None)
    return {
        "RMSLE":   rmsle(y_true, y_pred_c),
        "MAE":     mae(y_true, y_pred_c),
        "R2_log":  r2_log(y_true, y_pred_c),
        "MAE_med": float(np.median(np.abs(y_true - y_pred_c))),
    }

log = get_logger("evaluation")

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})
PALETTE = sns.color_palette("colorblind", 8)
MODEL_COLORS = {
    "XGBoost": PALETTE[0],
    "LightGBM": PALETTE[1],
    "RandomForest": PALETTE[2],
    "Ridge": PALETTE[3],
}


# ── Load artefacts ─────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = DATA_CLEAN / "test_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError("Run src/model.py first.")
    return pd.read_parquet(path)


def load_clean_data() -> pd.DataFrame:
    path = DATA_CLEAN / "ashrae_clean.parquet"
    return pd.read_parquet(path)


def load_model(name: str):
    return joblib.load(OUTPUTS_MODELS / f"{name.lower()}_model.joblib")


def load_feature_cols() -> list:
    path = OUTPUTS_MODELS / "feature_cols.json"
    if path.exists():
        return json.loads(path.read_text())
    return [c for c in FEATURE_COLS]


# ── Climate zone label reverse lookup ──────────────────────────────────────────

def decode_climate_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable climate zone label."""
    clean = load_clean_data()
    zone_map = (
        clean[["building_id", "climate_zone_enc"]]
        .drop_duplicates()
        .set_index("building_id")
    )
    # We need a mapping enc → label; reconstruct from SITE_CLIMATE_ZONE
    enc_vals = clean["climate_zone_enc"].unique()
    zone_labels = clean[["climate_zone_enc"]].copy()
    # Re-attach from clean data which still has site_id
    if "site_id" in clean.columns:
        clean2 = clean[["building_id", "site_id"]].drop_duplicates()
        clean2["climate_zone_label"] = clean2["site_id"].map(SITE_CLIMATE_ZONE)
        df = df.merge(clean2[["building_id", "climate_zone_label"]], on="building_id", how="left")
    else:
        df["climate_zone_label"] = "Unknown"
    return df


# ── Figure 1: Predicted vs Actual scatter by use group ──────────────────────

USE_GROUP_MAP = {
    0: "Education", 1: "Entertainment", 2: "Food Service", 3: "Healthcare",
    4: "Industrial", 5: "Lodging", 6: "Office", 7: "Other",
    8: "Parking", 9: "Public Assembly", 10: "Religious",
    11: "Retail", 12: "Services", 13: "Technology",
    14: "Utility", 15: "Warehouse",
}


def fig1_pred_vs_actual(preds: pd.DataFrame):
    log.info("Generating Figure 1: Predicted vs Actual (log scale) …")
    y_true = preds["y_true"].values
    y_pred = np.clip(preds["y_pred_XGBoost"].values, 0, None)
    use_enc = preds["use_group_enc"].values

    # Sample max 50k points for readability
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(50_000, len(y_true)), replace=False)
    yt_s, yp_s, ue_s = y_true[idx], y_pred[idx], use_enc[idx]

    use_labels = pd.Series(ue_s).map(USE_GROUP_MAP).fillna("Other")
    top6 = use_labels.value_counts().nlargest(6).index.tolist()
    use_labels = use_labels.where(use_labels.isin(top6), "Other")

    fig, ax = plt.subplots(figsize=(7, 6))
    palette6 = sns.color_palette("colorblind", len(use_labels.unique()))
    for i, grp in enumerate(sorted(use_labels.unique())):
        mask = (use_labels == grp).values
        ax.scatter(
            np.log1p(yt_s[mask]), np.log1p(yp_s[mask]),
            alpha=0.20, s=4, color=palette6[i % len(palette6)], label=grp,
        )

    lim = max(np.log1p(yt_s).max(), np.log1p(yp_s).max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="Perfect fit")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("log(1 + Actual Meter Reading)")
    ax.set_ylabel("log(1 + Predicted Meter Reading)")
    ax.set_title("Figure 1: XGBoost — Predicted vs. Actual (Log Scale, 50k sample)")
    ax.legend(markerscale=3, loc="upper left", framealpha=0.7)

    rmsle_val = rmsle(y_true, y_pred)
    r2l_val   = r2_log(y_true, y_pred)
    ax.text(0.97, 0.05,
            f"RMSLE = {rmsle_val:.4f}\nR²_log = {r2l_val:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    path = OUTPUTS_FIGURES / "fig1_pred_vs_actual.pdf"
    fig.savefig(path)
    plt.close(fig)
    log.info("  Saved: %s", path)


# ── Figure 2: SHAP global beeswarm ───────────────────────────────────────────

def fig2_shap_global(preds: pd.DataFrame):
    log.info("Generating Figure 2: SHAP global beeswarm …")
    feature_cols = load_feature_cols()
    model = load_model("XGBoost")

    # Use a sample for SHAP (max 5000 rows for speed)
    clean = load_clean_data()
    test_ids = preds["building_id"].unique()
    test_df = clean[clean["building_id"].isin(test_ids)]
    sample = test_df[feature_cols].sample(min(5000, len(test_df)), random_state=42)
    X_sample = sample.values.astype(np.float32)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Readable feature labels
    readable = {c: c.replace("_", " ").title() for c in feature_cols}
    readable["air_temperature"] = "Air Temp (°C)"
    readable["log_square_feet"] = "Log Floor Area"
    readable["dew_temperature"] = "Dew Temp (°C)"
    readable["temp_roll24h"] = "24h Rolling Temp"
    readable["HDD"] = "HDD"
    readable["CDD"] = "CDD"

    plt.figure(figsize=(8, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=[readable.get(c, c) for c in feature_cols],
        show=False, plot_type="dot", max_display=15,
    )
    plt.title("Figure 2: SHAP Global Feature Importance (XGBoost — 5,000 samples)")
    plt.tight_layout()
    path = OUTPUTS_FIGURES / "fig2_shap_global.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    log.info("  Saved: %s", path)

    # Save mean |SHAP| per feature as CSV for paper table
    mean_shap = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_shap.to_csv(OUTPUTS_TABLES / "shap_global_importance.csv", index=False)

    return shap_values, X_sample, feature_cols


# ── Figure 3: SHAP climate zone heatmap ─────────────────────────────────────

def fig3_shap_climate_heatmap(preds: pd.DataFrame, shap_values, X_sample, feature_cols):
    log.info("Generating Figure 3: SHAP climate zone heatmap …")
    clean = load_clean_data()
    test_ids = preds["building_id"].unique()
    test_df = clean[clean["building_id"].isin(test_ids)].copy()
    if "site_id" in test_df.columns:
        test_df["climate_zone_label"] = test_df["site_id"].map(SITE_CLIMATE_ZONE)
    else:
        test_df["climate_zone_label"] = "Unknown"

    sample = test_df[feature_cols + ["climate_zone_label"]].sample(
        min(5000, len(test_df)), random_state=42
    )

    top10_features = (
        pd.DataFrame({"feature": feature_cols,
                      "mean_shap": np.abs(shap_values).mean(axis=0)})
        .nlargest(10, "mean_shap")["feature"].tolist()
    )

    zones = sample["climate_zone_label"].unique()
    heatmap_data = {}
    for zone in zones:
        mask = (sample["climate_zone_label"] == zone).values
        if mask.sum() < 10:
            continue
        zone_shap = shap_values[mask]
        zone_mean = {
            f: float(np.abs(zone_shap[:, feature_cols.index(f)]).mean())
            for f in top10_features
        }
        heatmap_data[zone] = zone_mean

    hm_df = pd.DataFrame(heatmap_data, index=top10_features)
    hm_df.index = [c.replace("_", " ").title() for c in hm_df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        hm_df, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean |SHAP value|"},
    )
    ax.set_title("Figure 3: Mean |SHAP Value| per Feature Across Climate Zones")
    ax.set_xlabel("Climate Zone")
    ax.set_ylabel("Feature")
    plt.xticks(rotation=30, ha="right")

    path = OUTPUTS_FIGURES / "fig3_shap_climate_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    log.info("  Saved: %s", path)

    hm_df.to_csv(OUTPUTS_TABLES / "shap_climate_zone_heatmap.csv")


# ── Figure 4: Model comparison bar chart ─────────────────────────────────────

def fig4_model_comparison(preds: pd.DataFrame):
    log.info("Generating Figure 4: Model comparison (RMSLE / MAE / R²_log) …")
    model_names = ["XGBoost", "LightGBM", "RandomForest", "Ridge"]
    metrics_list = []
    y_true = preds["y_true"].values

    for name in model_names:
        col = f"y_pred_{name}"
        if col not in preds.columns:
            continue
        y_pred = np.clip(preds[col].values, 0, None)
        m = all_metrics(y_true, y_pred)
        m["Model"] = name
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    metrics_df.to_csv(OUTPUTS_TABLES / "model_comparison_full.csv")

    metric_labels = {
        "RMSLE":   "RMSLE (↓ better)",
        "MAE":     "MAE (kWh, ↓)",
        "R2_log":  "R² log-scale (↑)",
        "MAE_med": "Median AE (kWh, ↓)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()
    for ax, (metric, label) in zip(axes_flat, metric_labels.items()):
        vals = metrics_df[metric]
        colors = [MODEL_COLORS.get(m, "grey") for m in vals.index]
        bars = ax.bar(vals.index, vals.values, color=colors, edgecolor="black", linewidth=0.7,
                      width=0.55)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(y=0.18)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.suptitle(
        "Figure 4: Model Performance Comparison — Temporal Test Set (Oct–Dec)",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout(h_pad=3.0, w_pad=2.5)
    path = OUTPUTS_FIGURES / "fig4_model_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    log.info("  Saved: %s", path)
    return metrics_df


# ── Figure 5: SHAP dependence plots ──────────────────────────────────────────

def fig5_shap_dependence(preds: pd.DataFrame, shap_values, X_sample, feature_cols):
    log.info("Generating Figure 5: SHAP dependence plots …")
    feat1 = "air_temperature"
    feat2 = "log_square_feet"
    interaction = "CDD"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, feat in zip(axes, [feat1, feat2]):
        if feat not in feature_cols:
            log.warning("Feature %s not in feature_cols — skipping dependence plot", feat)
            continue
        fi = feature_cols.index(feat)
        shap_feat = shap_values[:, fi]
        x_feat = X_sample[:, fi]

        if interaction in feature_cols:
            ii = feature_cols.index(interaction)
            color_vals = X_sample[:, ii]
            sc = ax.scatter(x_feat, shap_feat, c=color_vals, cmap="RdYlBu_r",
                            alpha=0.3, s=5)
            plt.colorbar(sc, ax=ax, label=interaction.replace("_", " ").title())
        else:
            ax.scatter(x_feat, shap_feat, alpha=0.3, s=5, color=PALETTE[0])

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("SHAP Value")
        ax.set_title(f"SHAP Dependence: {feat.replace('_', ' ').title()}")

    fig.suptitle("Figure 5: SHAP Dependence Plots Coloured by CDD", y=1.02)
    plt.tight_layout()
    path = OUTPUTS_FIGURES / "fig5_shap_dependence.pdf"
    fig.savefig(path)
    plt.close(fig)
    log.info("  Saved: %s", path)


# ── LaTeX results table ───────────────────────────────────────────────────────

LATEX_TABLE_TEMPLATE = r"""
\begin{{table}}[H]
\caption{{Performance of machine learning models on the temporal test set (October--December 2016). 
RMSLE is the primary metric (ASHRAE competition standard, lower is better). 
R\textsuperscript{{2}}$_{{\log}}$ is computed on log-transformed values, robust to extreme outliers. Best values in \textbf{{bold}}.}}
\label{{tab:model-comparison}}
\centering
\setlength{{\tabcolsep}}{{8pt}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Model}} & \textbf{{RMSLE ($\downarrow$)}} & \textbf{{MAE (kWh, $\downarrow$)}} & \textbf{{R\textsuperscript{{2}}$_{{\log}}$ ($\uparrow$)}} & \textbf{{Median AE (kWh, $\downarrow$)}} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def write_latex_table(metrics_df: pd.DataFrame):
    rows = []
    best = {
        "RMSLE":   metrics_df["RMSLE"].min(),
        "MAE":     metrics_df["MAE"].min(),
        "R2_log":  metrics_df["R2_log"].max(),
        "MAE_med": metrics_df["MAE_med"].min(),
    }
    for model, row in metrics_df.iterrows():
        def fmt(col, val, higher_better=False):
            s = f"{val:.4f}" if col == "RMSLE" else f"{val:.2f}" if "MAE" in col else f"{val:.4f}"
            is_best = (val == best[col])
            return f"\\textbf{{{s}}}" if is_best else s
        rows.append(
            f"{model} & {fmt('RMSLE', row['RMSLE'])} & {fmt('MAE', row['MAE'])} "
            f"& {fmt('R2_log', row['R2_log'], True)} & {fmt('MAE_med', row['MAE_med'])} \\\\"
        )
    table = LATEX_TABLE_TEMPLATE.format(rows="\n".join(rows))
    path = OUTPUTS_TABLES / "table_model_comparison.tex"
    path.write_text(table.strip())
    log.info("LaTeX table saved: %s", path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log.info("=== Evaluation: Building Energy Consumption Prediction ===")

    preds = load_predictions()
    log.info("Test set: %d rows", len(preds))

    fig1_pred_vs_actual(preds)
    shap_values, X_sample, feature_cols = fig2_shap_global(preds)
    fig3_shap_climate_heatmap(preds, shap_values, X_sample, feature_cols)
    metrics_df = fig4_model_comparison(preds)
    fig5_shap_dependence(preds, shap_values, X_sample, feature_cols)
    write_latex_table(metrics_df)

    log.info("=== Evaluation complete. All figures in outputs/figures/ ===")


if __name__ == "__main__":
    main()
