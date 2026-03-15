"""
run_all.py — Full pipeline orchestrator for Building Energy Consumption Prediction.

Runs the complete research pipeline in order:
  1. Data collection  (Kaggle + UCI)
  2. Preprocessing    (clean, feature engineer, save parquet)
  3. Model training   (XGBoost, LightGBM, RF, Ridge)
  4. Evaluation       (metrics, SHAP, 5 figures, LaTeX tables)
  5. Copy figures     (outputs/figures → Paper/figures)

Usage:
    python run_all.py
    python run_all.py --skip-data   # skip download if data already present
"""

import sys
import shutil
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_step(label: str, module: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "-m", module.replace("/", ".").replace(".py", "")],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  ✓ {label} complete")


def copy_figures():
    """Copy generated figures from outputs/figures/ to Paper/figures/."""
    src_dir = ROOT / "outputs" / "figures"
    dst_dir = ROOT / "Paper" / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fig_file in src_dir.glob("fig*.pdf"):
        shutil.copy2(fig_file, dst_dir / fig_file.name)
        print(f"  Copied: {fig_file.name} → Paper/figures/")
        copied += 1

    if copied == 0:
        print("  [WARNING] No figure files found in outputs/figures/ — run evaluation first.")
    else:
        print(f"  ✓ {copied} figures copied to Paper/figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Run full Building Energy Consumption Prediction pipeline."
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download step (use if data already present)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Building Energy Consumption Prediction with SHAP Explainability")
    print("  Full Pipeline Run")
    print("="*60)

    if not args.skip_data:
        run_step("Data Collection", "src.data_collection")

    run_step("Preprocessing", "src.preprocessing")
    run_step("Model Training", "src.model")
    run_step("Evaluation & Figures", "src.evaluation")

    print(f"\n{'='*60}")
    print("  Copying figures to Paper/figures/")
    print(f"{'='*60}")
    copy_figures()

    print("\n" + "="*60)
    print("  ✅  PIPELINE COMPLETE")
    print("="*60)
    print("\nAll outputs:")
    print("  outputs/figures/    ← 5 publication figures (PDF)")
    print("  outputs/tables/     ← LaTeX table + SHAP CSVs")
    print("  outputs/models/     ← Trained model .joblib files")


if __name__ == "__main__":
    main()
