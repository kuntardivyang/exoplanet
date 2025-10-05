"""
Data preprocessing utilities for exoplanet detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple, List, Dict, Optional
import joblib
from pathlib import Path
from backend.config.config import (
    PROCESSED_DATA_DIR,
    MISSING_VALUE_THRESHOLD,
    TARGET_MAPPING
)
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExoplanetPreprocessor:
    """Comprehensive preprocessing pipeline for exoplanet data"""

    def __init__(self, target_column: str = 'koi_disposition'):
        self.target_column = target_column
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_columns = None
        self.selected_features = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by removing duplicates and irrelevant columns"""
        logger.info(f"Cleaning data. Initial shape: {df.shape}")

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Remove columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > MISSING_VALUE_THRESHOLD].index.tolist()

        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{MISSING_VALUE_THRESHOLD*100}% missing values")
            df = df.drop(columns=cols_to_drop)

        # Remove columns with constant values
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns")
            df = df.drop(columns=constant_cols)

        logger.info(f"Cleaned data shape: {df.shape}")
        return df

    def prepare_features_target(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable"""
        logger.info("Preparing features and target")

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Get target
        y = df[self.target_column].copy()

        # Get features
        if feature_columns is None:
            # Use all numeric columns except target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != self.target_column]

        # Remove ID and metadata columns
        metadata_keywords = ['id', 'name', 'rowid', 'row', 'date', 'comment', 'str', 'alias']
        feature_columns = [
            col for col in feature_columns
            if not any(keyword in col.lower() for keyword in metadata_keywords)
        ]

        self.feature_columns = feature_columns
        X = df[feature_columns].copy()

        logger.info(f"Features: {len(feature_columns)} columns, Target: {y.name}")
        return X, y

    def encode_target(self, y: pd.Series, mapping: Dict = None) -> np.ndarray:
        """Encode target variable"""
        logger.info(f"Encoding target variable. Unique values: {y.unique()}")

        if mapping is None:
            # Use label encoder
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
            else:
                y_encoded = self.label_encoder.transform(y)
        else:
            # Use custom mapping
            y_encoded = y.map(mapping).values
            logger.info(f"Applied custom mapping: {mapping}")

        return y_encoded

    def handle_missing_values(
        self,
        X: pd.DataFrame,
        strategy: str = 'median',
        use_knn: bool = False
    ) -> pd.DataFrame:
        """Handle missing values in features"""
        logger.info(f"Handling missing values. Strategy: {strategy}, KNN: {use_knn}")

        missing_before = X.isnull().sum().sum()
        logger.info(f"Missing values before imputation: {missing_before}")

        if missing_before == 0:
            logger.info("No missing values found")
            return X

        if use_knn:
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                X_imputed = pd.DataFrame(
                    self.imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X),
                    columns=X.columns,
                    index=X.index
                )
        else:
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy=strategy)
                X_imputed = pd.DataFrame(
                    self.imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X),
                    columns=X.columns,
                    index=X.index
                )

        missing_after = X_imputed.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")

        return X_imputed

    def handle_outliers(
        self,
        X: pd.DataFrame,
        method: str = 'clip',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Handle outliers in features"""
        logger.info(f"Handling outliers. Method: {method}")

        X_processed = X.copy()

        for col in X.columns:
            if method == 'clip':
                # Clip using IQR method
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                X_processed[col] = X[col].clip(lower, upper)

            elif method == 'zscore':
                # Clip using z-score
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
                X_processed[col] = X[col].clip(lower, upper)

        return X_processed

    def scale_features(
        self,
        X: pd.DataFrame,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """Scale features"""
        logger.info(f"Scaling features. Method: {method}")

        if self.scaler is None:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        return X_scaled

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = 'correlation',
        threshold: float = 0.01,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """Select most relevant features"""
        logger.info(f"Selecting features. Method: {method}")

        if method == 'correlation':
            # Calculate correlation with target
            correlations = {}
            for col in X.columns:
                try:
                    corr = np.corrcoef(X[col], y)[0, 1]
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except:
                    continue

            # Sort by correlation
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            if top_k:
                selected = [feat for feat, _ in sorted_features[:top_k]]
            else:
                selected = [feat for feat, corr in sorted_features if corr > threshold]

            self.selected_features = selected
            logger.info(f"Selected {len(selected)} features out of {len(X.columns)}")

            if len(selected) > 0:
                logger.info(f"Top 10 features by correlation: {dict(sorted_features[:10])}")

            return X[selected]

        return X

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        impute_strategy: str = 'median',
        scale_method: str = 'standard',
        handle_outliers: bool = True,
        select_features: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Complete preprocessing pipeline - fit and transform"""
        logger.info("Running full preprocessing pipeline (fit_transform)")

        # Clean data
        df_clean = self.clean_data(df)

        # Prepare features and target
        X, y = self.prepare_features_target(df_clean, feature_columns)

        # Encode target
        y_encoded = self.encode_target(y)

        # Handle missing values
        X = self.handle_missing_values(X, strategy=impute_strategy)

        # Handle outliers
        if handle_outliers:
            X = self.handle_outliers(X, method='clip')

        # Select features
        if select_features:
            X = self.select_features(X, y_encoded, threshold=0.01, top_k=50)

        # Scale features
        X = self.scale_features(X, method=scale_method)

        logger.info(f"Preprocessing complete. Final shape: X={X.shape}, y={y_encoded.shape}")

        return X, y_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        logger.info("Transforming new data using fitted preprocessor")

        if self.feature_columns is None:
            raise ValueError("Preprocessor not fitted yet. Call fit_transform first.")

        # Get features
        X = df[self.feature_columns].copy()

        # Handle missing values
        X = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns,
            index=X.index
        )

        # Select features
        if self.selected_features:
            X = X[self.selected_features]

        # Scale features
        X = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        return X

    def save(self, filepath: Path):
        """Save preprocessor to file"""
        logger.info(f"Saving preprocessor to {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path):
        """Load preprocessor from file"""
        logger.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)
