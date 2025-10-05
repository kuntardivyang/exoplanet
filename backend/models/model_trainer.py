"""
Model training pipeline for exoplanet detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, List, Optional
import joblib
from pathlib import Path
import json
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

from backend.config.config import MODEL_PARAMS, MODELS_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExoplanetModelTrainer:
    """Train and evaluate ML models for exoplanet detection"""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data_splits(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = TEST_SIZE,
        val_size: float = VALIDATION_SIZE
    ) -> Dict:
        """Split data into train, validation, and test sets"""
        logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }

    def initialize_models(self, custom_params: Optional[Dict] = None) -> Dict:
        """Initialize all available models"""
        logger.info("Initializing models")

        params = custom_params or MODEL_PARAMS
        models = {}

        # Random Forest
        models['random_forest'] = RandomForestClassifier(**params.get('random_forest', {}))

        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_params = params.get('xgboost', {})
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_params = params.get('lightgbm', {})
            models['lightgbm'] = lgb.LGBMClassifier(**lgb_params, verbose=-1)

        # Neural Network
        models['neural_network'] = MLPClassifier(**params.get('neural_network', {}))

        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state
        )

        self.models = models
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")

        return models

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[object, Dict]:
        """Train a single model"""
        logger.info(f"Training {model_name} model")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized")

        model = self.models[model_name]
        start_time = datetime.now()

        # Train model
        if X_val is not None and y_val is not None and model_name == 'xgboost':
            # XGBoost with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()

        # Training metrics
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        logger.info(f"{model_name} training completed in {training_time:.2f}s. Train accuracy: {train_accuracy:.4f}")

        return model, {
            'training_time': training_time,
            'train_accuracy': train_accuracy
        }

    def evaluate_model(
        self,
        model,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """Evaluate model performance"""
        logger.info(f"Evaluating {model_name} model")

        results = {'model_name': model_name}

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None

            results['validation'] = {
                'accuracy': float(accuracy_score(y_val, y_val_pred)),
                'precision': float(precision_score(y_val, y_val_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_val, y_val_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_val, y_val_pred, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist()
            }

            if y_val_proba is not None and len(np.unique(y_val)) == 2:
                results['validation']['roc_auc'] = float(roc_auc_score(y_val, y_val_proba[:, 1]))

        # Test metrics
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        results['test'] = {
            'accuracy': float(accuracy_score(y_test, y_test_pred)),
            'precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
        }

        if y_test_proba is not None and len(np.unique(y_test)) == 2:
            results['test']['roc_auc'] = float(roc_auc_score(y_test, y_test_proba[:, 1]))
            fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
            results['test']['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(
                X_test.columns,
                model.feature_importances_.tolist()
            ))
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            results['feature_importance'] = dict(sorted_importance[:20])

        logger.info(f"{model_name} - Test Accuracy: {results['test']['accuracy']:.4f}, F1: {results['test']['f1_score']:.4f}")

        return results

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict:
        """Train and evaluate all models"""
        logger.info("Training all models")

        all_results = {}

        for model_name in self.models.keys():
            try:
                # Train
                trained_model, train_info = self.train_model(
                    model_name, X_train, y_train, X_val, y_val
                )

                # Evaluate
                eval_results = self.evaluate_model(
                    trained_model, model_name, X_test, y_test, X_val, y_val
                )

                # Combine results
                all_results[model_name] = {
                    **train_info,
                    **eval_results
                }

                # Update stored model
                self.models[model_name] = trained_model

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        self.results = all_results

        # Find best model
        self._select_best_model()

        return all_results

    def _select_best_model(self):
        """Select best model based on validation F1 score"""
        if not self.results:
            logger.warning("No results available to select best model")
            return

        best_score = 0
        best_name = None

        for model_name, results in self.results.items():
            if 'validation' in results:
                f1 = results['validation']['f1_score']
            elif 'test' in results:
                f1 = results['test']['f1_score']
            else:
                continue

            if f1 > best_score:
                best_score = f1
                best_name = model_name

        if best_name:
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            logger.info(f"Best model: {best_name} (F1: {best_score:.4f})")

    def hyperparameter_tuning(
        self,
        model_name: str,
        param_grid: Dict,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict:
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info(f"Performing hyperparameter tuning for {model_name}")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized")

        base_model = self.models[model_name]

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }

    def save_model(self, model_name: str, filepath: Optional[Path] = None):
        """Save trained model to file"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        logger.info(f"Saving {model_name} model to {filepath}")
        joblib.dump(self.models[model_name], filepath)

        # Save results
        results_path = filepath.with_suffix('.json')
        if model_name in self.results:
            with open(results_path, 'w') as f:
                json.dump(self.results[model_name], f, indent=2)

        return filepath

    def save_all_models(self):
        """Save all trained models"""
        logger.info("Saving all models")
        saved_paths = {}

        for model_name in self.models.keys():
            try:
                path = self.save_model(model_name)
                saved_paths[model_name] = str(path)
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")

        return saved_paths

    def load_model(self, filepath: Path) -> object:
        """Load model from file"""
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models"""
        if not self.results:
            logger.warning("No results available for comparison")
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.results.items():
            row = {'model': model_name}

            if 'validation' in results:
                row['val_accuracy'] = results['validation']['accuracy']
                row['val_f1'] = results['validation']['f1_score']
                row['val_precision'] = results['validation']['precision']
                row['val_recall'] = results['validation']['recall']

            if 'test' in results:
                row['test_accuracy'] = results['test']['accuracy']
                row['test_f1'] = results['test']['f1_score']
                row['test_precision'] = results['test']['precision']
                row['test_recall'] = results['test']['recall']

            if 'training_time' in results:
                row['training_time'] = results['training_time']

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)
