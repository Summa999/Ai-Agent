import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw OHLCV data"""
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df = df.sort_index()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Add basic calculated fields
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['dollar_volume'] = df['close'] * df['volume']
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data"""
        # Forward fill for small gaps (up to 5 periods)
        df = df.fillna(method='ffill', limit=5)
        
        # For larger gaps, interpolate
        if df.isna().any().any():
            df = df.interpolate(method='time')
        
        # Drop rows with remaining NaN values
        df = df.dropna()
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, n_std: float = 5) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        # Calculate z-scores for returns
        if 'returns' in df.columns:
            z_scores = np.abs((df['returns'] - df['returns'].mean()) / df['returns'].std())
            df = df[z_scores < n_std]
        
        # Remove extreme price movements
        price_change = df['close'].pct_change()
        df = df[np.abs(price_change) < 0.2]  # Remove >20% price changes
        
        return df
    
    def resample_data(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe"""
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = df.resample(target_timeframe).agg(ohlc_dict)
        resampled.dropna(inplace=True)
        
        return resampled
    
    def align_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align data from multiple timeframes"""
        aligned_data = pd.DataFrame()
        
        # Get the highest timeframe as base
        base_timeframe = max(data_dict.keys())
        base_data = data_dict[base_timeframe].copy()
        
        # Add base data columns with timeframe suffix
        for col in base_data.columns:
            aligned_data[f"{col}_{base_timeframe}"] = base_data[col]
        
        # Align other timeframes
        for timeframe, data in data_dict.items():
            if timeframe != base_timeframe:
                # Resample to base timeframe
                resampled = self.resample_data(data, base_timeframe)
                
                # Add columns with timeframe suffix
                for col in resampled.columns:
                    aligned_data[f"{col}_{timeframe}"] = resampled[col]
        
        return aligned_data
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int, 
                        target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        if target_column:
            target = df[target_column].values
            features = df.drop(columns=[target_column]).values
        else:
            target = df.values
            features = df.values
        
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            if target_column:
                y.append(target[i + sequence_length])
            else:
                y.append(features[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """Normalize data"""
        normalized_df = df.copy()
        normalization_params = {}
        
        for column in df.columns:
            if column in ['open', 'high', 'low', 'close']:
                if method == 'minmax':
                    min_val = df[column].min()
                    max_val = df[column].max()
                    normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
                    normalization_params[column] = {'min': min_val, 'max': max_val}
                    
                elif method == 'zscore':
                    mean_val = df[column].mean()
                    std_val = df[column].std()
                    normalized_df[column] = (df[column] - mean_val) / std_val
                    normalization_params[column] = {'mean': mean_val, 'std': std_val}
        
        return normalized_df, normalization_params
    
    def denormalize_data(self, df: pd.DataFrame, normalization_params: Dict, 
                        method: str = 'minmax') -> pd.DataFrame:
        """Denormalize data"""
        denormalized_df = df.copy()
        
        for column, params in normalization_params.items():
            if column in df.columns:
                if method == 'minmax':
                    denormalized_df[column] = (df[column] * 
                                             (params['max'] - params['min']) + params['min'])
                elif method == 'zscore':
                    denormalized_df[column] = df[column] * params['std'] + params['mean']
        
        return denormalized_df