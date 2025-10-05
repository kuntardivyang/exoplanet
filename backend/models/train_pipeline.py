"""
Complete training pipeline for exoplanet detection models
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.utils.data_loader import DatasetLoader
from backend.utils.preprocessing import ExoplanetPreprocessor
from backend.models.model_trainer import ExoplanetModelTrainer
from backend.config.config import MODELS_DIR, PROCESSED_DATA_DIR, KEPLER_FEATURES
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class TrainingPipeline:
    """Complete pipeline for training exoplanet detection models"""

    def __init__(self, dataset: str = 'kepler'):
        self.dataset = dataset
        self.loader = DatasetLoader()
        self.preprocessor = ExoplanetPreprocessor(target_column='koi_disposition')
        self.trainer = ExoplanetModelTrainer()
        self.data = None
        self.splits = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def load_data(self):
        """Load dataset"""
        logger.info(f"Loading {self.dataset} dataset")

        if self.dataset == 'kepler':
            self.data = self.loader.load_kepler()
        elif self.dataset == 'tess':
            self.data = self.loader.load_tess()
        elif self.dataset == 'k2':
            self.data = self.loader.load_k2()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        logger.info(f"Loaded {len(self.data)} samples")
        return self.data

    def preprocess_data(self, save: bool = True):
        """Preprocess data"""
        logger.info("Preprocessing data")

        # For Kepler dataset, filter to confirmed and false positives only
        if self.dataset == 'kepler':
            logger.info("Filtering Kepler dataset to CONFIRMED and FALSE POSITIVE")
            initial_size = len(self.data)
            self.data = self.data[
                self.data['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])
            ].copy()
            logger.info(f"Filtered from {initial_size} to {len(self.data)} samples")

        # Preprocess
        X, y = self.preprocessor.fit_transform(
            self.data,
            feature_columns=None,  # Auto-select
            impute_strategy='median',
            scale_method='standard',
            handle_outliers=True,
            select_features=True
        )

        # Save preprocessor
        if save:
            preprocessor_path = PROCESSED_DATA_DIR / f'preprocessor_{self.dataset}_{self.timestamp}.pkl'
            self.preprocessor.save(preprocessor_path)
            logger.info(f"Saved preprocessor to {preprocessor_path}")

        return X, y

    def split_data(self, X: pd.DataFrame, y: np.ndarray):
        """Split data into train/val/test sets"""
        logger.info("Splitting data")
        self.splits = self.trainer.prepare_data_splits(X, y)
        return self.splits

    def train_models(self):
        """Train all models"""
        logger.info("Training models")

        if self.splits is None:
            raise ValueError("Data not split yet. Call split_data() first.")

        # Initialize models
        self.trainer.initialize_models()

        # Train all models
        results = self.trainer.train_all_models(
            self.splits['X_train'],
            self.splits['y_train'],
            self.splits['X_val'],
            self.splits['y_val'],
            self.splits['X_test'],
            self.splits['y_test']
        )

        return results

    def save_results(self):
        """Save training results and models"""
        logger.info("Saving results")

        # Save all models
        saved_models = self.trainer.save_all_models()

        # Save comparison
        comparison = self.trainer.get_model_comparison()
        comparison_path = MODELS_DIR / f'model_comparison_{self.dataset}_{self.timestamp}.csv'
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"Saved model comparison to {comparison_path}")

        # Save detailed results
        results_path = MODELS_DIR / f'training_results_{self.dataset}_{self.timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(self.trainer.results, f, indent=2)
        logger.info(f"Saved detailed results to {results_path}")

        # Save metadata
        metadata = {
            'dataset': self.dataset,
            'timestamp': self.timestamp,
            'num_samples': len(self.data),
            'num_features': len(self.preprocessor.selected_features or self.preprocessor.feature_columns),
            'feature_columns': self.preprocessor.selected_features or self.preprocessor.feature_columns,
            'best_model': self.trainer.best_model_name,
            'saved_models': saved_models,
            'data_splits': {
                'train': len(self.splits['y_train']),
                'val': len(self.splits['y_val']),
                'test': len(self.splits['y_test'])
            }
        }

        metadata_path = MODELS_DIR / f'metadata_{self.dataset}_{self.timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        return {
            'models': saved_models,
            'comparison': str(comparison_path),
            'results': str(results_path),
            'metadata': str(metadata_path)
        }

    def run_full_pipeline(self, save_results: bool = True):
        """Run the complete training pipeline"""
        logger.info("="*80)
        logger.info(f"Starting full training pipeline for {self.dataset} dataset")
        logger.info("="*80)

        # Load data
        self.load_data()

        # Preprocess
        X, y = self.preprocess_data(save=save_results)

        # Split data
        self.split_data(X, y)

        # Train models
        results = self.train_models()

        # Save results
        if save_results:
            saved_paths = self.save_results()
        else:
            saved_paths = {}

        # Print summary
        self.print_summary()

        logger.info("="*80)
        logger.info("Training pipeline completed successfully!")
        logger.info("="*80)

        return {
            'results': results,
            'saved_paths': saved_paths,
            'best_model': self.trainer.best_model_name
        }

    def print_summary(self):
        """Print training summary"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)

        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Total samples: {len(self.data)}")
        logger.info(f"Features used: {len(self.preprocessor.selected_features or self.preprocessor.feature_columns)}")

        if self.splits:
            logger.info(f"\nData splits:")
            logger.info(f"  Train: {len(self.splits['y_train'])} samples")
            logger.info(f"  Validation: {len(self.splits['y_val'])} samples")
            logger.info(f"  Test: {len(self.splits['y_test'])} samples")

        if self.trainer.results:
            logger.info(f"\nModel Results:")
            for model_name, results in self.trainer.results.items():
                logger.info(f"\n  {model_name.upper()}:")
                if 'test' in results:
                    logger.info(f"    Accuracy: {results['test']['accuracy']:.4f}")
                    logger.info(f"    Precision: {results['test']['precision']:.4f}")
                    logger.info(f"    Recall: {results['test']['recall']:.4f}")
                    logger.info(f"    F1 Score: {results['test']['f1_score']:.4f}")

        if self.trainer.best_model_name:
            logger.info(f"\nBest Model: {self.trainer.best_model_name}")

        logger.info("="*80 + "\n")


def main():
    """Main function to run training pipeline"""
    # Train on Kepler dataset
    pipeline = TrainingPipeline(dataset='kepler')
    results = pipeline.run_full_pipeline(save_results=True)

    return results


if __name__ == '__main__':
    main()
