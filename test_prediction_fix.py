#!/usr/bin/env python3
"""
Test the prediction fix for partial features
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.models.predictor import ExoplanetPredictor
import pandas as pd

# Find latest XGBoost model
models_dir = Path("data/models")
xgboost_models = sorted(models_dir.glob("xgboost_*.pkl"))

if not xgboost_models:
    print("❌ No XGBoost model found")
    sys.exit(1)

model_path = xgboost_models[-1]

# Find latest preprocessor
preprocessor_files = sorted(Path("data/processed").glob("preprocessor_*.pkl"))
if not preprocessor_files:
    print("❌ No preprocessor found")
    sys.exit(1)

preprocessor_path = preprocessor_files[-1]

print("="*60)
print("Testing Prediction with Partial Features")
print("="*60)
print(f"Model: {model_path.name}")
print(f"Preprocessor: {preprocessor_path.name}")
print()

# Initialize predictor
predictor = ExoplanetPredictor(
    model_path=str(model_path),
    preprocessor_path=str(preprocessor_path)
)
predictor.initialize()

# Test with minimal features (like from frontend)
test_sample = {
    'koi_period': 3.52474859,
    'koi_duration': 2.95750,
    'koi_depth': 2500.0,
    'koi_prad': 2.26,
    'koi_teq': 1370.0,
    'koi_insol': 93.59,
    'koi_model_snr': 35.8,
    'koi_steff': 6200.0,
    'koi_slogg': 4.18,
    'koi_srad': 1.793
}

print("Test Sample (10 features):")
for key, value in test_sample.items():
    print(f"  {key}: {value}")
print()

try:
    result = predictor.predict_single(test_sample)

    print("✅ PREDICTION SUCCESSFUL!")
    print()
    print(f"Prediction: {result.get('prediction', 'N/A')}")
    print(f"Predicted Label: {result.get('predicted_label', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")
    print(f"Probabilities: {result.get('probabilities', [])}")
    if 'class_probabilities' in result:
        print(f"Class Probabilities: {result['class_probabilities']}")
    print()

    label = result.get('predicted_label', result.get('prediction', ''))
    if 'CONFIRMED' in str(label).upper():
        print("🪐 This is likely a CONFIRMED exoplanet!")
    else:
        print("⚠️ This is likely a FALSE POSITIVE")

    print()
    print("="*60)
    print("✅ Fix successful! Predictions now work with partial features.")
    print("="*60)

except Exception as e:
    print(f"❌ PREDICTION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
