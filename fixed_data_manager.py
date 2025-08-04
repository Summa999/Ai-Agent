import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
import os

class FixedDataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_data(self, symbol='EURUSD', days=30):
        """Get data with proper error handling"""
        
        # Try different methods
        methods = [
            self._get_yfinance_data,
            self._get_generated_data,
            self._get_cached_data
        ]
        
        for method in methods:
            try:
                data = method(symbol, days)
                if data is not None and not data.empty:
                    # Ensure all required columns exist
                    data = self._ensure_columns(data)
                    return data
            except Exception as e:
                self.logger.warning(f"Method {method.__name__} failed: {e}")
                continue
        
        # If all methods fail, return generated data
        self.logger.warning("All data sources failed, using generated data")
        return self._get_generated_data(symbol, days)
    
    def _ensure_columns(self, df):
        """Ensure all required columns exist"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in df.columns and col.title() in df.columns:
                df[col] = df[col.title()]
            elif col not in df.columns:
                if col == 'volume':
                    # Generate fake volume for forex
                    df['volume'] = np.random.randint(1000, 10000, size=len(df))
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Keep only required columns
        df = df[required_columns]
        
        return df
    
    def _get_yfinance_data(self, symbol, days):
        """Get data from Yahoo Finance"""
        # Convert forex symbol
        yf_symbol = symbol
        if symbol == 'EURUSD':
            yf_symbol = 'EURUSD=X'
        elif symbol == 'GBPUSD':
            yf_symbol = 'GBPUSD=X'
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="5m")
        
        if df.empty:
            return None
            
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        return df
    
    def _get_generated_data(self, symbol, days):
        """Generate realistic fake data"""
        # Calculate number of 5-minute bars
        bars = days * 24 * 12  # 12 five-minute bars per hour
        
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='5min')
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 110.00
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate data
        np.random.seed(42)
        data = []
        price = base_price
        
        for date in dates:
            # Random walk
            change = np.random.randn() * 0.0005
            price = price * (1 + change)
            
            high = price * (1 + abs(np.random.randn() * 0.0002))
            low = price * (1 - abs(np.random.randn() * 0.0002))
            open_price = price * (1 + np.random.randn() * 0.0001)
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def _get_cached_data(self, symbol, days):
        """Get data from cache"""
        cache_file = f'data/cache/{symbol}_data.csv'
        
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Filter for requested days
            cutoff = datetime.now() - timedelta(days=days)
            return df[df.index >= cutoff]
        
        return None
    
    def save_to_cache(self, symbol, data):
        """Save data to cache"""
        os.makedirs('data/cache', exist_ok=True)
        cache_file = f'data/cache/{symbol}_data.csv'
        data.to_csv(cache_file)