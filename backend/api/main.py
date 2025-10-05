"""
FastAPI backend for Exoplanet Detection System
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.api.schemas import (
    PredictionRequest, BatchPredictionRequest, PredictionResponse,
    BatchPredictionResponse, ModelInfo, ModelListResponse,
    TrainingRequest, TrainingStatus, FeatureImportanceResponse,
    DatasetInfo, HealthCheck, ErrorResponse, FileUploadResponse
)
from backend.models.predictor import ExoplanetPredictor
from backend.models.train_pipeline import TrainingPipeline
from backend.utils.data_loader import DatasetLoader
from backend.config.config import (
    MODELS_DIR, CORS_ORIGINS, PROCESSED_DATA_DIR
)
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Detection API",
    description="AI-powered exoplanet detection and classification system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[ExoplanetPredictor] = None
training_status: Dict = {"status": "idle", "progress": 0, "message": None}


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    global predictor
    logger.info("Starting Exoplanet Detection API")

    # Try to load the latest model
    try:
        model_files = list(MODELS_DIR.glob("random_forest_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            preprocessor_files = list(PROCESSED_DATA_DIR.glob("preprocessor_*.pkl"))
            if preprocessor_files:
                latest_preprocessor = max(preprocessor_files, key=lambda p: p.stat().st_mtime)

                predictor = ExoplanetPredictor(latest_model, latest_preprocessor)
                predictor.initialize()
                logger.info(f"Loaded model: {latest_model.name}")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        logger.info("API will start without a loaded model. Train a model first.")


@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    return HealthCheck(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction on a single exoplanet candidate
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        result = predictor.predict_single(request.features)

        return PredictionResponse(
            prediction=result['prediction'],
            predicted_label=result.get('predicted_label'),
            confidence=result.get('confidence'),
            probabilities=result.get('probabilities'),
            class_probabilities=result.get('class_probabilities')
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions on multiple exoplanet candidates
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    try:
        results = predictor.predict_batch(request.samples)

        predictions = [
            PredictionResponse(
                prediction=r['prediction'],
                predicted_label=r.get('predicted_label'),
                confidence=r.get('confidence'),
                probabilities=r.get('probabilities'),
                class_probabilities=r.get('class_probabilities')
            )
            for r in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            num_samples=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/upload", response_model=FileUploadResponse)
async def predict_upload(file: UploadFile = File(...)):
    """
    Upload a CSV file and get predictions for all rows
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read CSV
        df = pd.read_csv(file.file)
        logger.info(f"Uploaded file: {file.filename}, rows: {len(df)}")

        # Convert to list of dicts
        samples = df.to_dict('records')

        # Get predictions
        results = predictor.predict_batch(samples)

        predictions = [
            PredictionResponse(
                prediction=r['prediction'],
                predicted_label=r.get('predicted_label'),
                confidence=r.get('confidence'),
                probabilities=r.get('probabilities'),
                class_probabilities=r.get('class_probabilities')
            )
            for r in results
        ]

        return FileUploadResponse(
            filename=file.filename,
            rows=len(df),
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available trained models"""
    try:
        model_files = list(MODELS_DIR.glob("*.pkl"))
        model_names = [f.stem for f in model_files]

        current = None
        if predictor and predictor.model_path:
            current = predictor.model_path.stem

        return ModelListResponse(
            models=model_names,
            current_model=current
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}/info", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    try:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Try to load metadata
        metadata_files = list(MODELS_DIR.glob("metadata_*.json"))
        metrics = None
        trained_date = None

        for meta_file in metadata_files:
            with open(meta_file) as f:
                meta = json.load(f)
                if model_name in meta.get('saved_models', {}).values():
                    trained_date = meta.get('timestamp')
                    break

        # Try to load results
        results_files = list(MODELS_DIR.glob("training_results_*.json"))
        for results_file in results_files:
            with open(results_file) as f:
                results = json.load(f)
                for model_type, model_results in results.items():
                    if model_type in model_name:
                        if 'test' in model_results:
                            metrics = {
                                'accuracy': model_results['test']['accuracy'],
                                'precision': model_results['test']['precision'],
                                'recall': model_results['test']['recall'],
                                'f1_score': model_results['test']['f1_score']
                            }
                        break

        return ModelInfo(
            model_name=model_name,
            model_type=model_name.split('_')[0] if '_' in model_name else "unknown",
            features_count=len(predictor.preprocessor.selected_features) if predictor else 0,
            classes=predictor.class_labels if predictor else None,
            trained_date=trained_date,
            metrics=metrics
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load/{model_name}")
async def load_model(model_name: str):
    """Load a specific model"""
    global predictor

    try:
        model_path = MODELS_DIR / f"{model_name}.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        # Find matching preprocessor
        preprocessor_files = list(PROCESSED_DATA_DIR.glob("preprocessor_*.pkl"))
        if not preprocessor_files:
            raise HTTPException(status_code=404, detail="No preprocessor found")

        latest_preprocessor = max(preprocessor_files, key=lambda p: p.stat().st_mtime)

        predictor = ExoplanetPredictor(model_path, latest_preprocessor)
        predictor.initialize()

        logger.info(f"Loaded model: {model_name}")
        return {"message": f"Model {model_name} loaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_task(dataset: str, model_types: Optional[List[str]] = None):
    """Background task for model training"""
    global training_status

    try:
        training_status = {"status": "training", "progress": 10, "message": "Loading data..."}

        pipeline = TrainingPipeline(dataset=dataset)
        pipeline.load_data()

        training_status["progress"] = 30
        training_status["message"] = "Preprocessing data..."

        X, y = pipeline.preprocess_data()

        training_status["progress"] = 50
        training_status["message"] = "Training models..."

        pipeline.split_data(X, y)
        results = pipeline.train_models()

        training_status["progress"] = 90
        training_status["message"] = "Saving models..."

        saved_paths = pipeline.save_results()

        training_status = {
            "status": "completed",
            "progress": 100,
            "message": "Training completed successfully",
            "results": {
                "best_model": pipeline.trainer.best_model_name,
                "saved_paths": saved_paths
            }
        }

        logger.info("Training task completed")

    except Exception as e:
        logger.error(f"Training error: {e}")
        training_status = {
            "status": "failed",
            "progress": 0,
            "message": f"Training failed: {str(e)}"
        }


@app.post("/train", response_model=TrainingStatus)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train new models on specified dataset
    """
    if training_status["status"] == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    background_tasks.add_task(run_training_task, request.dataset, request.model_types)

    return TrainingStatus(
        status="training",
        progress=0,
        message=f"Training started for {request.dataset} dataset"
    )


@app.get("/train/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(**training_status)


@app.get("/features/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(top_n: int = 20):
    """Get feature importance from the current model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        importance = predictor.get_feature_importance(top_n=top_n)
        return FeatureImportanceResponse(
            feature_importance=importance,
            top_n=top_n
        )
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List available datasets"""
    try:
        loader = DatasetLoader()
        datasets = []

        # Kepler
        try:
            kepler_df = loader.load_kepler()
            datasets.append(DatasetInfo(
                name="kepler",
                rows=len(kepler_df),
                columns=len(kepler_df.columns),
                features=kepler_df.columns.tolist()[:20],
                target_distribution=kepler_df['koi_disposition'].value_counts().to_dict() if 'koi_disposition' in kepler_df.columns else None
            ))
        except:
            pass

        # TESS
        try:
            tess_df = loader.load_tess()
            datasets.append(DatasetInfo(
                name="tess",
                rows=len(tess_df),
                columns=len(tess_df.columns),
                features=tess_df.columns.tolist()[:20]
            ))
        except:
            pass

        # K2
        try:
            k2_df = loader.load_k2()
            datasets.append(DatasetInfo(
                name="k2",
                rows=len(k2_df),
                columns=len(k2_df.columns),
                features=k2_df.columns.tolist()[:20]
            ))
        except:
            pass

        return datasets
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/{dataset_name}/sample")
async def get_dataset_sample(dataset_name: str, n: int = 10):
    """Get sample rows from a dataset"""
    try:
        loader = DatasetLoader()

        if dataset_name == 'kepler':
            df = loader.load_kepler()
        elif dataset_name == 'tess':
            df = loader.load_tess()
        elif dataset_name == 'k2':
            df = loader.load_k2()
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")

        sample = df.sample(min(n, len(df))).to_dict('records')
        return {"dataset": dataset_name, "sample": sample, "total_rows": len(df)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from backend.config.config import API_HOST, API_PORT

    uvicorn.run(
        "backend.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
