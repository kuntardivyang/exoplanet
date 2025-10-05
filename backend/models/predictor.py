"""
Prediction service for exoplanet detection
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import joblib
from backend.utils.logger import setup_logger
from backend.utils.preprocessing import ExoplanetPreprocessor

logger = setup_logger(__name__)


class ExoplanetPredictor:
    """Make predictions on new exoplanet candidates"""

    def __init__(self, model_path: Path, preprocessor_path: Path):
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.class_labels = None

    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

    def load_preprocessor(self):
        """Load preprocessor"""
        logger.info(f"Loading preprocessor from {self.preprocessor_path}")
        self.preprocessor = joblib.load(self.preprocessor_path)

    def initialize(self):
        """Initialize predictor by loading model and preprocessor"""
        self.load_model()
        self.load_preprocessor()

        # Try to get class labels
        if hasattr(self.preprocessor, 'label_encoder') and self.preprocessor.label_encoder:
            self.class_labels = self.preprocessor.label_encoder.classes_.tolist()
            logger.info(f"Class labels: {self.class_labels}")

    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        return_proba: bool = True
    ) -> Dict:
        """
        Make predictions on new data

        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            return_proba: Whether to return probability scores

        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first. Call initialize()")

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a DataFrame, dict, or list of dicts")

        logger.info(f"Making predictions on {len(df)} samples")

        try:
            # Preprocess data
            X = self.preprocessor.transform(df)

            # Make predictions
            predictions = self.model.predict(X)

            results = {
                'predictions': predictions.tolist(),
                'num_samples': len(df)
            }

            # Add class labels if available
            if self.class_labels:
                predicted_labels = [self.class_labels[pred] for pred in predictions]
                results['predicted_labels'] = predicted_labels

            # Add probabilities
            if return_proba and hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                results['probabilities'] = probabilities.tolist()

                # Add confidence (max probability)
                confidence = np.max(probabilities, axis=1)
                results['confidence'] = confidence.tolist()

            logger.info(f"Predictions completed: {np.bincount(predictions)}")

            return results

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_single(self, data: Dict) -> Dict:
        """
        Make prediction on a single sample

        Args:
            data: Dictionary with feature values

        Returns:
            Dictionary with prediction and probability
        """
        results = self.predict(data, return_proba=True)

        single_result = {
            'prediction': results['predictions'][0]
        }

        if 'predicted_labels' in results:
            single_result['predicted_label'] = results['predicted_labels'][0]

        if 'probabilities' in results:
            single_result['probabilities'] = results['probabilities'][0]
            single_result['confidence'] = results['confidence'][0]

            # Add class probabilities
            if self.class_labels:
                class_probs = dict(zip(self.class_labels, results['probabilities'][0]))
                single_result['class_probabilities'] = class_probs

        return single_result

    def predict_batch(self, data_list: List[Dict]) -> List[Dict]:
        """
        Make predictions on a batch of samples

        Args:
            data_list: List of dictionaries with feature values

        Returns:
            List of prediction results
        """
        results = self.predict(data_list, return_proba=True)

        batch_results = []
        for i in range(results['num_samples']):
            result = {
                'prediction': results['predictions'][i]
            }

            if 'predicted_labels' in results:
                result['predicted_label'] = results['predicted_labels'][i]

            if 'probabilities' in results:
                result['probabilities'] = results['probabilities'][i]
                result['confidence'] = results['confidence'][i]

                if self.class_labels:
                    class_probs = dict(zip(self.class_labels, results['probabilities'][i]))
                    result['class_probabilities'] = class_probs

            batch_results.append(result)

        return batch_results

    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """Get feature importance from the model"""
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return {}

        feature_names = self.preprocessor.selected_features or self.preprocessor.feature_columns
        importances = self.model.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        feature_importance = {
            feature_names[i]: float(importances[i])
            for i in indices
        }

        return feature_importance

    def explain_prediction(self, data: Dict) -> Dict:
        """
        Explain a prediction by showing feature contributions

        Args:
            data: Dictionary with feature values

        Returns:
            Dictionary with prediction explanation
        """
        result = self.predict_single(data)

        # Add feature importance
        result['feature_importance'] = self.get_feature_importance()

        # Add input features
        result['input_features'] = data

        return result
