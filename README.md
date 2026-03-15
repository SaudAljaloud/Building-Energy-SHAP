# Building Energy Consumption Prediction with SHAP Explainability

A fully reproducible machine learning pipeline for predicting commercial building energy consumption across multiple climate zones, with cross-climate-zone SHAP explainability analysis.

**Paper:** *Building Energy Consumption Prediction Using Gradient Boosting with SHAP Explainability: A Multi-Climate-Zone and Multi-Building-Type Analysis* — submitted to MDPI Applied Sciences.

---

## Overview

This repository benchmarks four machine learning models on the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction) dataset (20.2 million hourly meter readings, 1,449 commercial buildings, 16 sites). The key contribution is a **cross-climate-zone SHAP attribution analysis** that reveals how feature importance varies across Hot & Dry, Hot & Humid, Mixed, and Cold climate zones.

**Best result:** XGBoost — RMSLE = 1.1944, MAE = 368.1 kWh, R²_log = 0.677 (temporal holdout: Oct–Dec test)

---

## Folder Structure

```
building-energy-shap/
├── src/
│   ├── utils.py            ← Paths, metrics, feature definitions
│   ├── data_collection.py  ← Download ASHRAE (Kaggle) + UCI datasets
│   ├── preprocessing.py    ← Clean, feature-engineer, save parquet
│   ├── model.py            ← Train 4 models with temporal split
│   └── evaluation.py       ← Metrics, SHAP, 5 publication figures
├── notebooks/
│   ├── 01_exploration.ipynb  ← EDA: distributions, temporal patterns
│   └── 02_results.ipynb      ← Figures, cross-building-type analysis
├── outputs/
│   ├── figures/            ← 5 publication figures (PDF, 300 dpi)
│   ├── tables/             ← LaTeX table + SHAP CSVs
│   └── models/             ← Trained model binaries (gitignored)
├── data/                   ← Raw and clean data (gitignored)
├── run_all.py              ← One-command full pipeline
├── requirements.txt
└── .gitignore
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/building-energy-shap.git
cd building-energy-shap
pip install -r requirements.txt

# 2. Configure Kaggle credentials (one-time)
#    Register at https://www.kaggle.com
#    Accept competition rules: https://www.kaggle.com/c/ashrae-energy-prediction
#    Download your API token → place at ~/.kaggle/kaggle.json

# 3. Run the full pipeline
python run_all.py

# 4. Or run steps individually
python src/data_collection.py   # Download data
python src/preprocessing.py     # Clean + feature engineer
python src/model.py             # Train models
python src/evaluation.py        # Generate figures and tables

# 5. Skip download if data already present
python run_all.py --skip-data
```

---

## Data Sources

| Dataset | Source | Access | Size |
|---|---|---|---|
| ASHRAE GPED III | [Kaggle Competition](https://www.kaggle.com/c/ashrae-energy-prediction) | Free (account + rules acceptance required) | ~2.5 GB |
| UCI Energy Efficiency | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) | Direct download | 45 KB |

> **Note:** Raw data is not included in this repository. Follow the Kaggle setup instructions above.

---

## Models

| Model | Type | RMSLE | MAE (kWh) | R²_log |
|---|---|---|---|---|
| **XGBoost** | Gradient Boosting | **1.1944** | **368.1** | **0.677** |
| LightGBM | Gradient Boosting | 1.2210 | 401.3 | 0.651 |
| Random Forest | Ensemble | 1.5170 | 612.4 | 0.423 |
| Ridge Regression | Linear | 1.8960 | 891.2 | 0.187 |

Temporal holdout: trained on January–September 2016, tested on October–December 2016.

---

## Publication Figures

| Figure | Description |
|---|---|
| Fig 1 | Predicted vs Actual scatter (log scale) coloured by building use group |
| Fig 2 | SHAP global beeswarm — top 15 features (XGBoost) |
| Fig 3 | SHAP mean\|value\| heatmap: features × climate zones |
| Fig 4 | Model comparison: RMSLE, MAE, R²_log, Median AE (2×2 panels) |
| Fig 5 | SHAP dependence plots: air temperature and log floor area |

---

## Key Findings

1. **Log floor area** is the dominant energy predictor across all five climate zones (mean |SHAP| = 0.788)
2. **Seasonality** (day-of-year) is 2.7× more important in Hot & Dry than Cold climates
3. **Building age** is 2.3× more important in Hot & Dry zones — consistent with envelope degradation under high solar loads
4. **Meter type** importance is 2.0× higher in Cold zones, consistent with district heating prevalence
5. **Air temperature** exhibits a U-shaped SHAP profile: neutral zone 10–22°C, positive SHAP at both extremes (heating + cooling demand)

---

## Feature Engineering

29 features engineered from three raw ASHRAE files:

- **Temporal (8):** hour, day-of-week, month, day-of-year, weekend flag, US holiday flag, sine/cosine cyclical encodings
- **Building (5):** log floor area, building age, floor count, primary use encoding, use-group encoding
- **Weather (11):** air temperature, dew point, cloud coverage, wind speed/direction, pressure, precipitation, relative humidity (Magnus formula), HDD, CDD (base 18°C), 24h rolling mean temperature
- **Identifiers (5):** meter type, site ID, climate zone encoding, use-group, wind chill index

---

## Requirements

Python 3.10+. See `requirements.txt` for full list. Key dependencies:

```
xgboost>=1.7.6
lightgbm>=4.0.0
shap>=0.42.1
scikit-learn>=1.3.0
pandas>=2.0.0
kaggle>=1.5.16
```

---

## Citation

If you use this code in your research, please cite the associated paper (citation details will be added upon publication).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
