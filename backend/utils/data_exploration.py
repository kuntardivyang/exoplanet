"""
Data exploration and analysis utilities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from backend.utils.logger import setup_logger
from backend.config.config import LOGS_DIR

logger = setup_logger(__name__)
plt.style.use('seaborn-v0_8-darkgrid')


class DataExplorer:
    """Explore and analyze exoplanet datasets"""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name

    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        logger.info(f"Generating summary report for {self.name}")

        report = {
            'basic_info': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
            },
            'data_types': self.df.dtypes.value_counts().to_dict(),
            'missing_data': self._analyze_missing_data(),
            'numeric_summary': self._get_numeric_summary(),
            'categorical_summary': self._get_categorical_summary(),
            'duplicates': self.df.duplicated().sum()
        }

        return report

    def _analyze_missing_data(self) -> Dict:
        """Analyze missing data patterns"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100)

        missing_df = pd.DataFrame({
            'count': missing,
            'percentage': missing_pct
        })
        missing_df = missing_df[missing_df['count'] > 0].sort_values('count', ascending=False)

        return {
            'total_missing': int(missing.sum()),
            'columns_with_missing': len(missing_df),
            'top_missing': missing_df.head(10).to_dict()
        }

    def _get_numeric_summary(self) -> Dict:
        """Get summary statistics for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}

        stats = self.df[numeric_cols].describe()
        return {
            'count': len(numeric_cols),
            'columns': numeric_cols.tolist(),
            'statistics': stats.to_dict()
        }

    def _get_categorical_summary(self) -> Dict:
        """Get summary for categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return {}

        summary = {}
        for col in categorical_cols:
            summary[col] = {
                'unique_values': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(5).to_dict()
            }

        return {
            'count': len(categorical_cols),
            'columns': categorical_cols.tolist(),
            'details': summary
        }

    def detect_outliers(self, column: str, method: str = 'iqr') -> Dict:
        """Detect outliers in a numeric column"""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        if not np.issubdtype(self.df[column].dtype, np.number):
            raise ValueError(f"Column '{column}' is not numeric")

        data = self.df[column].dropna()

        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'method': method,
            'total_outliers': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'bounds': {
                'lower': float(lower_bound) if method == 'iqr' else None,
                'upper': float(upper_bound) if method == 'iqr' else None
            }
        }

    def correlation_analysis(self, columns: List[str] = None, threshold: float = 0.5) -> Dict:
        """Analyze correlations between numeric features"""
        numeric_df = self.df.select_dtypes(include=[np.number])

        if columns:
            numeric_df = numeric_df[columns]

        corr_matrix = numeric_df.corr()

        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr
        }

    def feature_importance_analysis(self, target_col: str, top_n: int = 20) -> Dict:
        """Analyze feature importance using correlation with target"""
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        correlations = {}
        for col in numeric_cols:
            try:
                corr = self.df[col].corr(self.df[target_col])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        return {
            'top_features': dict(sorted_corr[:top_n]),
            'all_correlations': correlations
        }

    def get_class_balance(self, target_col: str) -> Dict:
        """Analyze class balance for classification"""
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        value_counts = self.df[target_col].value_counts()
        percentages = (value_counts / len(self.df) * 100)

        return {
            'counts': value_counts.to_dict(),
            'percentages': percentages.to_dict(),
            'is_balanced': percentages.max() / percentages.min() < 2 if len(percentages) > 1 else True,
            'imbalance_ratio': float(percentages.max() / percentages.min()) if len(percentages) > 1 else 1.0
        }
