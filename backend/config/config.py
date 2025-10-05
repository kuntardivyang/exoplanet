"""
Configuration settings for the Exoplanet Detection System
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Log directory
LOGS_DIR = BASE_DIR / "logs"

# Dataset files
KEPLER_FILE = RAW_DATA_DIR / "cumulative_2025.10.04_10.17.04.csv"
TESS_FILE = RAW_DATA_DIR / "TOI_2025.10.04_10.17.23.csv"
K2_FILE = RAW_DATA_DIR / "k2pandc_2025.10.04_10.17.32.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Feature engineering settings
FEATURE_SELECTION_THRESHOLD = 0.01  # Correlation threshold
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with >50% missing

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'neural_network': {
        'hidden_layer_sizes': (128, 64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 32,
        'learning_rate': 'adaptive',
        'max_iter': 500,
        'random_state': RANDOM_STATE
    }
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# Kepler-specific features to use
KEPLER_FEATURES = [
    # Orbital parameters
    'koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq',

    # Stellar parameters
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass',

    # Transit parameters
    'koi_impact', 'koi_dor',

    # Signal to noise
    'koi_model_snr',

    # False positive flags
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
]

# Target column mapping
TARGET_MAPPING = {
    'CONFIRMED': 1,
    'CANDIDATE': 0,
    'FALSE POSITIVE': -1
}

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
