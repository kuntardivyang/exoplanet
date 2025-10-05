#!/usr/bin/env python3
"""
Test script to verify the complete exoplanet detection system
Tests: Models, API, Predictions, Light Curves, and Chat functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import json
from backend.models.predictor import ExoplanetPredictor
from backend.utils.data_loader import DatasetLoader
from backend.utils.light_curve_loader import LightCurveLoader, LightCurveFeatureExtractor

API_URL = "http://localhost:8000"

def test_models_exist():
    """Test 1: Check if models are trained and saved"""
    print("\n" + "="*60)
    print("TEST 1: Checking Trained Models")
    print("="*60)

    models_dir = Path("data/models")
    model_files = list(models_dir.glob("*.pkl"))

    print(f"✓ Found {len(model_files)} model files")

    # Check for latest models
    model_types = ['xgboost', 'lightgbm', 'random_forest', 'neural_network', 'gradient_boosting']
    for model_type in model_types:
        latest = sorted(models_dir.glob(f"{model_type}_*.pkl"))
        if latest:
            print(f"  ✓ {model_type}: {latest[-1].name}")
        else:
            print(f"  ✗ {model_type}: NOT FOUND")

    return len(model_files) > 0

def test_prediction():
    """Test 2: Test prediction functionality"""
    print("\n" + "="*60)
    print("TEST 2: Testing Predictions")
    print("="*60)

    try:
        # Find latest XGBoost model (best performer)
        models_dir = Path("data/models")
        xgboost_models = sorted(models_dir.glob("xgboost_*.pkl"))

        if not xgboost_models:
            print("✗ No XGBoost model found")
            return False

        model_path = xgboost_models[-1]

        # Find corresponding preprocessor
        timestamp = model_path.stem.split('_', 1)[1]
        preprocessor_path = Path(f"data/processed/preprocessor_kepler_{timestamp}.pkl")

        if not preprocessor_path.exists():
            # Try to find any preprocessor
            preprocessors = sorted(Path("data/processed").glob("preprocessor_*.pkl"))
            if preprocessors:
                preprocessor_path = preprocessors[-1]
            else:
                print("✗ No preprocessor found")
                return False

        print(f"Loading model: {model_path.name}")
        print(f"Loading preprocessor: {preprocessor_path.name}")

        # Initialize predictor
        predictor = ExoplanetPredictor(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path)
        )
        predictor.initialize()

        # Test sample (values from Kepler dataset)
        test_sample = {
            'koi_period': 3.52474859,
            'koi_time0bk': 170.538750,
            'koi_impact': 0.146,
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

        result = predictor.predict_single(test_sample)

        print(f"\n✓ Prediction successful!")
        print(f"  Classification: {result['classification']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities: {result['probabilities']}")

        return True

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def test_light_curves():
    """Test 3: Test light curve analysis"""
    print("\n" + "="*60)
    print("TEST 3: Testing Light Curve Analysis")
    print("="*60)

    try:
        lightcurve_dir = Path("lightcurve")
        tbl_files = list(lightcurve_dir.glob("*.tbl"))

        if not tbl_files:
            print("✗ No .tbl files found in lightcurve directory")
            return False

        print(f"✓ Found {len(tbl_files)} light curve files")

        # Test parsing one file
        loader = LightCurveLoader()
        test_file = tbl_files[0]

        metadata, data = loader.load_light_curve(test_file)
        print(f"\n✓ Parsed: {test_file.name}")
        print(f"  Object: {metadata.get('star_id', 'Unknown')}")
        print(f"  Data points: {len(data)}")
        print(f"  Columns: {list(data.columns)}")

        # Test feature extraction
        extractor = LightCurveFeatureExtractor()
        features = extractor.extract_all_features(data, metadata)

        print(f"\n✓ Extracted {len(features)} features:")
        for key, value in list(features.items())[:5]:
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"✗ Light curve analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_health():
    """Test 4: Test API health"""
    print("\n" + "="*60)
    print("TEST 4: Testing API Connection")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy!")
            print(f"  Status: {data.get('status')}")
            print(f"  Models loaded: {data.get('models_loaded')}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API (is it running?)")
        print("  Start with: python3 backend/api/main.py")
        return False
    except Exception as e:
        print(f"✗ API health check failed: {e}")
        return False

def test_chat_system():
    """Test 5: Test Chat/RAG system"""
    print("\n" + "="*60)
    print("TEST 5: Testing Chat System")
    print("="*60)

    try:
        # Check if Ollama is available
        try:
            ollama_check = requests.get("http://localhost:11434/api/version", timeout=2)
            if ollama_check.status_code == 200:
                print("✓ Ollama is running")
            else:
                print("⚠ Ollama may not be running properly")
        except:
            print("⚠ Ollama not detected (optional for chat)")
            print("  Install: curl -fsSL https://ollama.com/install.sh | sh")
            print("  Start: ollama serve")
            print("  Pull model: ollama pull llama3.2")
            return False

        # Test chat status endpoint
        response = requests.get(f"{API_URL}/chat/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Chat system status:")
            print(f"  LLM available: {data.get('llm_available')}")
            print(f"  Vector store: {data.get('vector_store_status')}")
            print(f"  Documents indexed: {data.get('documents_count')}")
            return True
        else:
            print(f"✗ Chat status check failed: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to chat API")
        return False
    except Exception as e:
        print(f"⚠ Chat system test inconclusive: {e}")
        return False

def test_data_loading():
    """Test 6: Test dataset loading"""
    print("\n" + "="*60)
    print("TEST 6: Testing Dataset Loading")
    print("="*60)

    try:
        loader = DatasetLoader()

        # Test Kepler
        kepler_df = loader.load_kepler()
        print(f"✓ Kepler dataset: {len(kepler_df)} samples, {len(kepler_df.columns)} features")

        # Test TESS
        tess_df = loader.load_tess()
        print(f"✓ TESS dataset: {len(tess_df)} samples, {len(tess_df.columns)} features")

        # Test K2
        k2_df = loader.load_k2()
        print(f"✓ K2 dataset: {len(k2_df)} samples, {len(k2_df.columns)} features")

        return True

    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("EXOPLANET DETECTION SYSTEM - COMPREHENSIVE TEST")
    print("="*60)

    results = {
        "Models Exist": test_models_exist(),
        "Prediction": test_prediction(),
        "Light Curves": test_light_curves(),
        "Dataset Loading": test_data_loading(),
        "API Health": test_api_health(),
        "Chat System": test_chat_system(),
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed! System is fully functional.")
    elif passed >= total - 1:
        print("\n⚠ Most tests passed. Check failed tests above.")
    else:
        print("\n❌ Multiple tests failed. Review the output above.")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Start Backend:  python3 backend/api/main.py")
    print("2. Start Frontend: cd frontend && npm run dev")
    print("3. Start Ollama:   ollama serve  (in separate terminal)")
    print("4. Visit: http://localhost:3000")
    print("="*60)

if __name__ == "__main__":
    main()
