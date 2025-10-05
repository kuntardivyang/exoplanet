"""
Light Curve Data Loader for .tbl files
Parses exoplanet transit light curves from table format
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class LightCurveLoader:
    """Load and parse .tbl light curve files"""

    def __init__(self, data_dir: Path = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / "lightcurve"
        else:
            self.data_dir = Path(data_dir)

    def parse_metadata(self, filepath: Path) -> Dict:
        """Extract metadata from .tbl file header"""
        metadata = {}

        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('\\'):
                    # Parse metadata line
                    match = re.match(r'\\(\w+)\s*=\s*"([^"]*)"', line)
                    if match:
                        key, value = match.groups()
                        # Try to convert to appropriate type
                        try:
                            if '.' in value:
                                metadata[key.lower()] = float(value)
                            else:
                                metadata[key.lower()] = int(value) if value.isdigit() else value
                        except:
                            metadata[key.lower()] = value
                elif line.startswith('|'):
                    # We've reached the data section
                    break

        return metadata

    def parse_data(self, filepath: Path) -> pd.DataFrame:
        """Parse time-series data from .tbl file"""
        # Find where data starts
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find the line with column headers
        header_line = None
        data_start = None

        for i, line in enumerate(lines):
            if line.startswith('|') and 'HJD' in line:
                header_line = i
                # Data starts 3 lines after header (header, types, units)
                data_start = i + 3
                break

        if header_line is None:
            raise ValueError(f"Could not find data header in {filepath}")

        # Extract column names
        header = lines[header_line]
        columns = [col.strip() for col in header.split('|') if col.strip()]

        # Read data
        data_lines = [line.strip() for line in lines[data_start:] if line.strip()]

        # Parse data rows
        data = []
        for line in data_lines:
            values = line.split()
            if len(values) == len(columns):
                data.append([float(v) if v.replace('.', '').replace('-', '').isdigit() else v
                            for v in values])

        df = pd.DataFrame(data, columns=columns)

        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        return df

    def load_light_curve(self, filepath: Path) -> Tuple[Dict, pd.DataFrame]:
        """Load both metadata and data from a light curve file"""
        logger.info(f"Loading light curve: {filepath.name}")

        metadata = self.parse_metadata(filepath)
        data = self.parse_data(filepath)

        # Add filepath to metadata
        metadata['filepath'] = str(filepath)
        metadata['filename'] = filepath.name

        logger.info(f"Loaded {len(data)} data points for {metadata.get('star_id', 'unknown')}")

        return metadata, data

    def load_all_light_curves(self) -> List[Tuple[Dict, pd.DataFrame]]:
        """Load all .tbl files in the directory"""
        tbl_files = sorted(self.data_dir.glob("*.tbl"))
        logger.info(f"Found {len(tbl_files)} light curve files")

        light_curves = []
        for filepath in tbl_files:
            try:
                metadata, data = self.load_light_curve(filepath)
                light_curves.append((metadata, data))
            except Exception as e:
                logger.error(f"Error loading {filepath.name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(light_curves)} light curves")
        return light_curves

    def get_by_star_id(self, star_id: str) -> List[Tuple[Dict, pd.DataFrame]]:
        """Get all light curves for a specific star"""
        all_curves = self.load_all_light_curves()
        return [(meta, data) for meta, data in all_curves
                if meta.get('star_id', '').upper() == star_id.upper()]

    def get_unique_stars(self) -> List[str]:
        """Get list of unique star IDs in the dataset"""
        all_curves = self.load_all_light_curves()
        stars = set(meta.get('star_id', 'UNKNOWN') for meta, _ in all_curves)
        return sorted(list(stars))

    def get_dataset_summary(self) -> pd.DataFrame:
        """Generate summary statistics for all light curves"""
        all_curves = self.load_all_light_curves()

        summaries = []
        for metadata, data in all_curves:
            summary = {
                'star_id': metadata.get('star_id', 'UNKNOWN'),
                'filename': metadata.get('filename'),
                'n_points': len(data),
                'period': metadata.get('period'),
                'min_flux': data['relative_flux'].min() if 'relative_flux' in data.columns else None,
                'max_flux': data['relative_flux'].max() if 'relative_flux' in data.columns else None,
                'transit_depth': 1.0 - data['relative_flux'].min() if 'relative_flux' in data.columns else None,
                'duration_days': data['hjd'].max() - data['hjd'].min() if 'hjd' in data.columns else None,
            }
            summaries.append(summary)

        return pd.DataFrame(summaries)


class LightCurveFeatureExtractor:
    """Extract features from light curve time-series"""

    @staticmethod
    def extract_transit_features(data: pd.DataFrame, metadata: Dict) -> Dict:
        """Extract features describing the transit"""
        features = {}

        if 'relative_flux' in data.columns:
            flux = data['relative_flux'].values

            # Basic statistics
            features['mean_flux'] = np.mean(flux)
            features['std_flux'] = np.std(flux)
            features['min_flux'] = np.min(flux)
            features['max_flux'] = np.max(flux)

            # Transit depth (normalized)
            features['transit_depth'] = 1.0 - np.min(flux)
            features['transit_depth_ppm'] = (1.0 - np.min(flux)) * 1e6

            # Flux variation
            features['flux_range'] = np.max(flux) - np.min(flux)
            features['flux_variation'] = np.std(flux) / np.mean(flux)

        if 'phase' in data.columns:
            phase = data['phase'].values
            flux = data['relative_flux'].values

            # Find transit (lowest flux region)
            transit_mask = flux < (np.mean(flux) - 2 * np.std(flux))
            if np.any(transit_mask):
                transit_phases = phase[transit_mask]
                features['transit_phase_start'] = np.min(transit_phases)
                features['transit_phase_end'] = np.max(transit_phases)
                features['transit_duration_phase'] = np.max(transit_phases) - np.min(transit_phases)

        # Metadata features
        if 'period' in metadata:
            features['orbital_period'] = float(metadata['period'])

        if 'number_of_points' in metadata:
            features['n_datapoints'] = int(metadata['number_of_points'])

        return features

    @staticmethod
    def extract_statistical_features(data: pd.DataFrame) -> Dict:
        """Extract statistical features from light curve"""
        features = {}

        if 'relative_flux' in data.columns:
            flux = data['relative_flux'].values

            # Moments
            features['skewness'] = pd.Series(flux).skew()
            features['kurtosis'] = pd.Series(flux).kurtosis()

            # Percentiles
            for p in [5, 25, 50, 75, 95]:
                features[f'flux_p{p}'] = np.percentile(flux, p)

            # Autocorrelation (first lag)
            if len(flux) > 1:
                features['autocorr_lag1'] = pd.Series(flux).autocorr(lag=1)

        if 'relative_flux_uncertainty' in data.columns:
            err = data['relative_flux_uncertainty'].values
            features['mean_uncertainty'] = np.mean(err)
            features['median_uncertainty'] = np.median(err)

        return features

    @staticmethod
    def extract_frequency_features(data: pd.DataFrame) -> Dict:
        """Extract frequency domain features using FFT"""
        features = {}

        if 'relative_flux' in data.columns and len(data) >= 10:
            flux = data['relative_flux'].values

            # Remove mean
            flux_centered = flux - np.mean(flux)

            # FFT
            fft = np.fft.fft(flux_centered)
            power = np.abs(fft) ** 2
            freqs = np.fft.fftfreq(len(flux_centered))

            # Get positive frequencies only
            positive_mask = freqs > 0
            power_positive = power[positive_mask]
            freqs_positive = freqs[positive_mask]

            if len(power_positive) > 0:
                # Dominant frequency
                dominant_idx = np.argmax(power_positive)
                features['dominant_frequency'] = freqs_positive[dominant_idx]
                features['dominant_power'] = power_positive[dominant_idx]

                # Total power
                features['total_power'] = np.sum(power_positive)

                # Power in low frequencies (< 0.1)
                low_freq_mask = freqs_positive < 0.1
                if np.any(low_freq_mask):
                    features['low_freq_power'] = np.sum(power_positive[low_freq_mask])

        return features

    def extract_all_features(self, data: pd.DataFrame, metadata: Dict) -> Dict:
        """Extract all features from a light curve"""
        features = {}

        # Transit features
        transit_features = self.extract_transit_features(data, metadata)
        features.update({f'transit_{k}': v for k, v in transit_features.items()})

        # Statistical features
        stat_features = self.extract_statistical_features(data)
        features.update({f'stat_{k}': v for k, v in stat_features.items()})

        # Frequency features
        freq_features = self.extract_frequency_features(data)
        features.update({f'freq_{k}': v for k, v in freq_features.items()})

        # Add star ID
        features['star_id'] = metadata.get('star_id', 'UNKNOWN')

        return features


if __name__ == "__main__":
    # Test the loader
    loader = LightCurveLoader()

    # Get summary
    summary = loader.get_dataset_summary()
    print("\nDataset Summary:")
    print(summary)

    # Get unique stars
    stars = loader.get_unique_stars()
    print(f"\nUnique stars: {stars}")

    # Load one light curve and extract features
    metadata, data = loader.load_light_curve(
        loader.data_dir / "UID_0007562_data_AXA_001.tbl"
    )

    extractor = LightCurveFeatureExtractor()
    features = extractor.extract_all_features(data, metadata)

    print(f"\nExtracted features for {metadata['star_id']}:")
    for key, value in features.items():
        print(f"  {key}: {value}")
