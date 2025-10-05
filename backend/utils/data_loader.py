"""
Data loading utilities for exoplanet datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from backend.config.config import KEPLER_FILE, TESS_FILE, K2_FILE
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class DatasetLoader:
    """Load and manage exoplanet datasets"""

    def __init__(self):
        self.kepler_df = None
        self.tess_df = None
        self.k2_df = None

    def load_kepler(self) -> pd.DataFrame:
        """Load Kepler dataset"""
        logger.info(f"Loading Kepler dataset from {KEPLER_FILE}")
        try:
            self.kepler_df = pd.read_csv(KEPLER_FILE, comment='#')
            logger.info(f"Loaded Kepler dataset: {self.kepler_df.shape[0]} rows, {self.kepler_df.shape[1]} columns")

            # Basic info
            if 'koi_disposition' in self.kepler_df.columns:
                disposition_counts = self.kepler_df['koi_disposition'].value_counts()
                logger.info(f"Kepler dispositions:\n{disposition_counts}")

            return self.kepler_df
        except Exception as e:
            logger.error(f"Error loading Kepler dataset: {e}")
            raise

    def load_tess(self) -> pd.DataFrame:
        """Load TESS dataset"""
        logger.info(f"Loading TESS dataset from {TESS_FILE}")
        try:
            self.tess_df = pd.read_csv(TESS_FILE, comment='#')
            logger.info(f"Loaded TESS dataset: {self.tess_df.shape[0]} rows, {self.tess_df.shape[1]} columns")

            # Basic info
            if 'tfopwg_disp' in self.tess_df.columns:
                disposition_counts = self.tess_df['tfopwg_disp'].value_counts()
                logger.info(f"TESS dispositions:\n{disposition_counts}")

            return self.tess_df
        except Exception as e:
            logger.error(f"Error loading TESS dataset: {e}")
            raise

    def load_k2(self) -> pd.DataFrame:
        """Load K2 dataset"""
        logger.info(f"Loading K2 dataset from {K2_FILE}")
        try:
            self.k2_df = pd.read_csv(K2_FILE, comment='#')
            logger.info(f"Loaded K2 dataset: {self.k2_df.shape[0]} rows, {self.k2_df.shape[1]} columns")

            return self.k2_df
        except Exception as e:
            logger.error(f"Error loading K2 dataset: {e}")
            raise

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets"""
        logger.info("Loading all datasets...")
        return {
            'kepler': self.load_kepler(),
            'tess': self.load_tess(),
            'k2': self.load_k2()
        }

    def get_dataset_info(self, df: pd.DataFrame, name: str) -> Dict:
        """Get comprehensive dataset information"""
        info = {
            'name': name,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }

        # Add statistics for numeric columns
        numeric_stats = df.select_dtypes(include=[np.number]).describe().to_dict()
        info['numeric_stats'] = numeric_stats

        return info

    def analyze_target_distribution(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze target variable distribution"""
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found in dataset")
            return {}

        distribution = df[target_col].value_counts().to_dict()
        percentages = (df[target_col].value_counts(normalize=True) * 100).to_dict()

        return {
            'counts': distribution,
            'percentages': percentages,
            'total': len(df),
            'unique_values': df[target_col].nunique()
        }
