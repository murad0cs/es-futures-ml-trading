"""
Centralized path configuration for the trading system
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Specific subdirectories
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"
PHASE1_MODELS_DIR = MODELS_DIR / "phase1"
PHASE2_MODELS_DIR = MODELS_DIR / "phase2"
PHASE1_RESULTS_DIR = RESULTS_DIR / "phase1"
PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"

# Create subdirectories
for sub_dir in [SYNTHETIC_DATA_DIR, HISTORICAL_DATA_DIR, 
                 PHASE1_MODELS_DIR, PHASE2_MODELS_DIR,
                 PHASE1_RESULTS_DIR, PHASE2_RESULTS_DIR]:
    sub_dir.mkdir(parents=True, exist_ok=True)

def get_data_file(filename: str, data_type: str = "synthetic") -> Path:
    """Get path for data file"""
    if data_type == "synthetic":
        return SYNTHETIC_DATA_DIR / filename
    else:
        return HISTORICAL_DATA_DIR / filename

def get_model_file(filename: str, phase: int = 1) -> Path:
    """Get path for model file"""
    if phase == 1:
        return PHASE1_MODELS_DIR / filename
    else:
        return PHASE2_MODELS_DIR / filename

def get_results_file(filename: str, phase: int = 1) -> Path:
    """Get path for results file"""
    if phase == 1:
        return PHASE1_RESULTS_DIR / filename
    else:
        return PHASE2_RESULTS_DIR / filename

def get_report_file(filename: str) -> Path:
    """Get path for report file"""
    return REPORTS_DIR / filename