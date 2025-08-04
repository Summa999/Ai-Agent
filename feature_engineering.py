import numpy as np
import pandas as pd
import ta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineering:
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)
        self.feature_names = []
        
    def create_price_features(self, df):
        """Create advanced price-based features"""
        # Basic returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['squared_returns'] = df['returns'] ** 2
        
        # Multiple timeframe returns
        for period in [5, 10, 20, 50]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Volatility measures
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_50'] = df['returns'].rolling(50).std()
        df['volatility_ratio'] = df['volatility_20'] / (df['volatility_50'] + 1e-10)
        
        # Price efficiency
        df['efficiency_ratio'] = abs(df['close'].diff(10)) / (df['close'].diff().abs().rolling(10).sum() + 1e-10)
        
        return df
    
    def create_technical_indicators(self, df):
        """Create comprehensive technical indicators"""
        # Trend Indicators
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], period)
            
            # Price relative to MA
            df[f'close_to_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
            df[f'close_to_ema_{period}'] = df['close'] / (df[f'ema_{period}'] + 1e-10)
        
        # MACD variations
        try:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        except:
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_diff'] = 0
        
        # ADX
        try:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
        except:
            df['adx'] = 14
            df['adx_pos'] = 0
            df['adx_neg'] = 0
        
        # Momentum Indicators
        # RSI variations
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        # Stochastic
        try:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
        except:
            df['stoch_k'] = 50
            df['stoch_d'] = 50
        
        # CCI
        try:
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        except:
            df['cci'] = 0
        
        # MFI
        try:
            df['mfi'] = ta.volume.MFIIndicator(
                df['high'], df['low'], df['close'], df['volume']
            ).money_flow_index()
        except:
            df['mfi'] = 50
        
        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
        
        # Volatility Indicators
        # Bollinger Bands
        for period in [10, 20, 30]:
            try:
                bb = ta.volatility.BollingerBands(df['close'], window=period)
                df[f'bb_high_{period}'] = bb.bollinger_hband()
                df[f'bb_low_{period}'] = bb.bollinger_lband()
                df[f'bb_mid_{period}'] = bb.bollinger_mavg()
                df[f'bb_width_{period}'] = bb.bollinger_wband()
                df[f'bb_pband_{period}'] = bb.bollinger_pband()
            except:
                df[f'bb_high_{period}'] = df['close'] * 1.02
                df[f'bb_low_{period}'] = df['close'] * 0.98
                df[f'bb_mid_{period}'] = df['close']
                df[f'bb_width_{period}'] = 0.04
                df[f'bb_pband_{period}'] = 0.5
        
        # ATR
        for period in [7, 14, 21]:
            try:
                df[f'atr_{period}'] = ta.volatility.AverageTrueRange(
                    df['high'], df['low'], df['close'], window=period
                ).average_true_range()
            except:
                df[f'atr_{period}'] = (df['high'] - df['low']).rolling(period).mean()
        
        # Volume Indicators
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        
        # OBV
        try:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        except:
            df['obv'] = df['volume'].cumsum()
        
        # VWAP
        try:
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume']
            ).volume_weighted_average_price()
        except:
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df
    
    def create_pattern_features(self, df):
        """Create candlestick pattern features"""
        # Candlestick patterns
        df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)) < 0.1
        df['hammer'] = ((df['high'] - df['low']) > 3 * abs(df['open'] - df['close'])) & \
                       ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) > 0.6) & \
                       ((df['open'] - df['low']) / (df['high'] - df['low'] + 1e-10) > 0.6)
        
        df['shooting_star'] = ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) & \
                              ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10) > 0.6) & \
                              ((df['high'] - df['open']) / (df['high'] - df['low'] + 1e-10) > 0.6)
        
        # Price action patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        
        # Support and Resistance
        for period in [20, 50]:
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'sr_ratio_{period}'] = (df['close'] - df[f'support_{period}']) / \
                                        (df[f'resistance_{period}'] - df[f'support_{period}'] + 1e-10)
        
        # Convert boolean to int
        bool_columns = ['doji', 'hammer', 'shooting_star', 'higher_high', 'lower_low', 'inside_bar']
        for col in bool_columns:
            df[col] = df[col].astype(int)
        
        return df
    
    def create_microstructure_features(self, df):
        """Create market microstructure features"""
        # Spread analysis
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / (df['close'] + 1e-10)
        df['avg_spread_20'] = df['spread'].rolling(20).mean()
        df['spread_volatility'] = df['spread'].rolling(20).std()
        
        # Tick analysis
        df['tick_direction'] = np.sign(df['close'].diff())
        df['tick_streak'] = df['tick_direction'].groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()
        ).cumsum()
        
        # Order flow imbalance proxy
        df['close_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['volume_direction'] = df['volume'] * np.where(df['close'] > df['open'], 1, -1)
        df['cumulative_volume_delta'] = df['volume_direction'].rolling(20).sum()
        
        # Liquidity measures
        df['amihud_illiquidity'] = abs(df['returns']) / (df['volume'] + 1e-10)
        df['roll_measure'] = 2 * np.sqrt(abs(df['returns'].rolling(2).cov(df['returns'].shift(1)).fillna(0)))
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical features"""
        # Distribution features
        for period in [20, 50]:
            df[f'skewness_{period}'] = df['returns'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['returns'].rolling(period).kurt()
            
        # Entropy
        def entropy(x):
            if len(x) < 10:
                return 0
            p, _ = np.histogram(x.dropna(), bins=10)
            p = p / (p.sum() + 1e-10)
            return -np.sum(p * np.log(p + 1e-10))
        
        df['entropy_20'] = df['returns'].rolling(20).apply(entropy)
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_{lag}'] = df['returns'].rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else 0
            )
        
        return df
    
    def create_regime_features(self, df):
        """Create market regime features"""
        # Trend strength
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / (df['sma_50'] + 1e-10)
        
        # Market regime
        df['bull_market'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['high_volatility'] = (df['volatility_20'] > df['volatility_20'].rolling(100).mean()).astype(int)
        
        # Trend consistency
        df['trend_consistency'] = df['returns'].rolling(20).apply(
            lambda x: sum(x > 0) / (len(x) + 1e-10)
        )
        
        return df
    
    def create_interaction_features(self, df):
        """Create feature interactions"""
        # Technical indicator interactions
        df['rsi_bb_signal'] = ((df['rsi_14'] < 30) & (df['close'] < df['bb_low_20'])).astype(int)
        df['macd_rsi_divergence'] = (
            (df['macd_diff'] > 0) & (df['rsi_14'] < 50)
        ).astype(int)
        
        # Volume-price interactions
        df['volume_price_correlation'] = df['close'].rolling(20).corr(df['volume'])
        df['high_volume_breakout'] = (
            (df['volume'] > df['volume_sma_20'] * 1.5) & 
            (df['close'] > df['resistance_20'])
        ).astype(int)
        
        # Momentum-volatility interaction
        df['momentum_volatility_ratio'] = df['roc_10'] / (df['atr_14'] + 1e-10)
        
        return df
    
    def create_cyclical_features(self, df):
        """Create time-based cyclical features"""
        # Check if we have datetime index or time column
        if 'time' in df.columns:
            dt_series = pd.to_datetime(df['time'])
        elif isinstance(df.index, pd.DatetimeIndex):
            dt_series = df.index
        else:
            # Skip cyclical features if no time information
            return df
        
        # Time features
        df['hour'] = dt_series.hour
        df['day_of_week'] = dt_series.dayofweek
        df['day_of_month'] = dt_series.day
        df['month'] = dt_series.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Trading session features
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def create_all_features(self, df):
        """Create all features"""
        df = df.copy()
        
        # Handle MT5 column naming
        if 'tick_volume' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['tick_volume']
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create all feature groups
        df = self.create_price_features(df)
        df = self.create_technical_indicators(df)
        df = self.create_pattern_features(df)
        df = self.create_microstructure_features(df)
        df = self.create_statistical_features(df)
        df = self.create_regime_features(df)
        df = self.create_interaction_features(df)
        df = self.create_cyclical_features(df)
        
        # Remove NaN values
        df = df.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5).dropna()
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in 
                             ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']]
        
        return df
    
    def select_features(self, df, target_col, n_features=100):
        """Select most important features using various methods"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # Prepare data
        X = df[self.feature_names]
        y = df[target_col]
        
        # Method 1: Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(self.feature_names)))
        mi_selector.fit(X, y)
        mi_scores = dict(zip(self.feature_names, mi_selector.scores_))
        
        # Method 2: Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = dict(zip(self.feature_names, rf.feature_importances_))
        
        # Method 3: Correlation with target
        correlations = {}
        for col in self.feature_names:
            correlations[col] = abs(df[col].corr(y))
        
        # Combine scores
        combined_scores = {}
        for feature in self.feature_names:
            combined_scores[feature] = (
                mi_scores.get(feature, 0) * 0.4 +
                rf_importance.get(feature, 0) * 0.4 +
                correlations.get(feature, 0) * 0.2
            )
        
        # Select top features
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in top_features[:n_features]]
        
        return selected_features, combined_scores