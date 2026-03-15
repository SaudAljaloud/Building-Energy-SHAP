"""
Data collection — Building Energy Consumption Prediction.

Sources:
  1. ASHRAE Great Energy Predictor III (Kaggle) — primary dataset
     kaggle competitions download -c ashrae-energy-prediction
  2. UCI Energy Efficiency Dataset — cross-dataset validation
     https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx

Usage:
    python src/data_collection.py
"""

import os
import zipfile
import subprocess
import requests
from pathlib import Path
from src.utils import DATA_RAW, get_logger

log = get_logger("data_collection")


# ── ASHRAE Dataset ─────────────────────────────────────────────────────────────

def download_ashrae_via_kaggle() -> bool:
    """
    Download ASHRAE GPED III using the Kaggle CLI.
    Requires ~/.kaggle/kaggle.json (or KAGGLE_USERNAME + KAGGLE_KEY env vars).
    Returns True if successful.
    """
    ashrae_dir = DATA_RAW / "ashrae"
    ashrae_dir.mkdir(exist_ok=True)

    required_files = [
        "train.csv",
        "building_metadata.csv",
        "weather_train.csv",
    ]

    already_present = all((ashrae_dir / f).exists() for f in required_files)
    if already_present:
        log.info("ASHRAE files already present — skipping download.")
        return True

    log.info("Downloading ASHRAE GPED III from Kaggle …")
    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "ashrae-energy-prediction",
                "-p", str(ashrae_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            log.error("Kaggle CLI error: %s", result.stderr)
            return False
        log.info("Kaggle download complete. Extracting …")
        _extract_zips(ashrae_dir)
        return True
    except FileNotFoundError:
        log.error(
            "kaggle CLI not found. Install with: pip install kaggle\n"
            "Then place kaggle.json at ~/.kaggle/kaggle.json"
        )
        return False
    except subprocess.TimeoutExpired:
        log.error("Kaggle download timed out.")
        return False


def _extract_zips(directory: Path) -> None:
    for zip_path in directory.glob("*.zip"):
        log.info("Extracting %s …", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)
        zip_path.unlink()


# ── UCI Energy Efficiency Dataset ─────────────────────────────────────────────

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
)

UCI_COLUMNS = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}


def download_uci_energy_efficiency() -> bool:
    """Download UCI Building Energy Efficiency dataset (no auth required)."""
    uci_path = DATA_RAW / "uci_energy_efficiency.xlsx"
    if uci_path.exists():
        log.info("UCI dataset already present — skipping download.")
        return True

    log.info("Downloading UCI Energy Efficiency dataset …")
    try:
        resp = requests.get(UCI_URL, timeout=60)
        resp.raise_for_status()
        uci_path.write_bytes(resp.content)
        log.info("UCI dataset saved to %s", uci_path)
        return True
    except requests.RequestException as exc:
        log.error("Failed to download UCI dataset: %s", exc)
        return False


# ── Verification helpers ───────────────────────────────────────────────────────

def verify_downloads() -> dict:
    """Return download status for each required file."""
    status = {}
    ashrae_dir = DATA_RAW / "ashrae"
    for fname in ["train.csv", "building_metadata.csv", "weather_train.csv"]:
        path = ashrae_dir / fname
        status[f"ashrae/{fname}"] = {
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else 0,
        }
    uci_path = DATA_RAW / "uci_energy_efficiency.xlsx"
    status["uci_energy_efficiency.xlsx"] = {
        "exists": uci_path.exists(),
        "size_mb": round(uci_path.stat().st_size / 1e6, 3) if uci_path.exists() else 0,
    }
    return status


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log.info("=== Data Collection: Building Energy Consumption Prediction ===")

    ashrae_ok = download_ashrae_via_kaggle()
    uci_ok = download_uci_energy_efficiency()

    status = verify_downloads()
    log.info("Download summary:")
    for fname, info in status.items():
        marker = "✓" if info["exists"] else "✗"
        log.info("  %s %s  (%.1f MB)", marker, fname, info["size_mb"])

    if not ashrae_ok:
        log.warning(
            "\nASHRAE dataset missing. Ensure kaggle.json is configured:\n"
            "  1. Register at https://www.kaggle.com\n"
            "  2. Accept competition rules: https://www.kaggle.com/c/ashrae-energy-prediction\n"
            "  3. Download API token → place at ~/.kaggle/kaggle.json\n"
            "  4. Run: pip install kaggle && python src/data_collection.py"
        )

    if not uci_ok:
        log.warning("UCI dataset missing — cross-dataset validation will be skipped.")

    log.info("=== Data collection complete ===")


if __name__ == "__main__":
    main()
