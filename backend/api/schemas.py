"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    features: Dict[str, float] = Field(..., description="Dictionary of feature values")
    return_probabilities: bool = Field(True, description="Whether to return probability scores")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    samples: List[Dict[str, float]] = Field(..., description="List of feature dictionaries")
    return_probabilities: bool = Field(True, description="Whether to return probability scores")


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: int
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[List[float]] = None
    class_probabilities: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    num_samples: int
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Model information schema"""
    model_name: str
    model_type: str
    features_count: int
    classes: Optional[List[str]] = None
    trained_date: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    models: List[str]
    current_model: Optional[str] = None


class TrainingRequest(BaseModel):
    """Request schema for training"""
    dataset: str = Field("kepler", description="Dataset to use (kepler, tess, k2)")
    model_types: Optional[List[str]] = Field(None, description="Specific models to train")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Custom hyperparameters")


class TrainingStatus(BaseModel):
    """Training status response"""
    status: str = Field(..., description="Status: training, completed, failed")
    progress: Optional[float] = Field(None, description="Progress percentage")
    message: Optional[str] = None
    results: Optional[Dict] = None


class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    feature_importance: Dict[str, float]
    top_n: int


class DatasetInfo(BaseModel):
    """Dataset information"""
    name: str
    rows: int
    columns: int
    features: List[str]
    target_distribution: Optional[Dict[str, int]] = None


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    model_loaded: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class FileUploadResponse(BaseModel):
    """File upload response"""
    filename: str
    rows: int
    predictions: List[PredictionResponse]
    timestamp: datetime = Field(default_factory=datetime.now)
