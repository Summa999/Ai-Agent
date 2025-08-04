import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import ruptures as rpt

class MarketRegimeDetector:
    def __init__(self):
        self.regime_model = None
        self.scaler = StandardScaler()
        self.current_regime = None
        self.regime_history = []
        
    def detect_regime(self, data):
        """Detect current market regime"""
        features = self.prepare_regime_features(data)
        
        # Hidden Markov Model for regime detection
        regime = self.hmm_regime_detection(features)
        
        # Change point detection
        change_points = self.detect_change_points(data)
        
        # Volatility regime
        vol_regime = self.detect_volatility_regime(data)
        
        # Trend regime
        trend_regime = self.detect_trend_regime(data)
        
        # Combine all regime indicators
        combined_regime = self.combine_regimes(regime, vol_regime, trend_regime)
        
        self.current_regime = combined_regime
        self.regime_history.append({
            'timestamp': data.index[-1],
            'regime': combined_regime,
            'hmm_regime': regime,
            'vol_regime': vol_regime,
            'trend_regime': trend_regime,
            'change_points': change_points
        })
        
        return combined_regime
    
    def prepare_regime_features(self, data):
        """Prepare features for regime detection"""
        features = pd.DataFrame(index=data.index)
        
        # Returns
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility
        features['realized_vol'] = features['returns'].rolling(20).std()
        features['garch_vol'] = self.calculate_garch_volatility(features['returns'])
        
        # Volume
        features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Momentum
        features['momentum_10'] = data['close'].pct_change(10)
        features['momentum_30'] = data['close'].pct_change(30)
        
        # Market microstructure
        features['spread'] = (data['high'] - data['low']) / data['close']
        
        return features.dropna()
    
    def hmm_regime_detection(self, features, n_regimes=4):
        """Hidden Markov Model for regime detection"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        model.fit(scaled_features)
        
        # Predict regimes
        regimes = model.predict(scaled_features)
        
        # Get regime probabilities
        regime_probs = model.predict_proba(scaled_features)
        
        # Current regime
        current_regime = regimes[-1]
        
        # Regime characteristics
        regime_means = model.means_
        regime_covars = model.covars_
        
        return {
            'current': current_regime,
            'history': regimes,
            'probabilities': regime_probs[-1],
            'characteristics': {
                'means': regime_means,
                'covariances': regime_covars
            }
        }
    
    def detect_change_points(self, data):
        """Detect structural breaks in time series"""
        # Use multiple change point detection methods
        
        # Method 1: Pelt
        signal = data['close'].values
        algo = rpt.Pelt(model="rbf").fit(signal)
        change_points_pelt = algo.predict(pen=10)
        
        # Method 2: Binary Segmentation
        algo_binseg = rpt.Binseg(model="l2").fit(signal)
        change_points_binseg = algo_binseg.predict(n_bkps=5)
        
        # Method 3: Window sliding
        algo_window = rpt.Window(width=50, model="l2").fit(signal)
        change_points_window = algo_window.predict(n_bkps=5)
        
        # Combine and validate change points
        all_change_points = set(change_points_pelt + change_points_binseg + change_points_window)
        
        # Filter out points too close to each other
        filtered_points = []
        for point in sorted(all_change_points):
            if not filtered_points or point - filtered_points[-1] > 20:
                filtered_points.append(point)
        
        return filtered_points
    
    def detect_volatility_regime(self, data):
        """Detect volatility regime"""
        returns = data['close'].pct_change().dropna()
        
        # Calculate various volatility measures
        vol_measures = pd.DataFrame()
        vol_measures['realized_vol_10'] = returns.rolling(10).std()
        vol_measures['realized_vol_30'] = returns.rolling(30).std()
        vol_measures['high_low_vol'] = ((data['high'] - data['low']) / data['close']).rolling(20).mean()
        
        # Current volatility percentile
        current_vol = vol_measures['realized_vol_10'].iloc[-1]
        vol_percentile = (current_vol > vol_measures['realized_vol_10']).mean()
        
        # Classify regime
        if vol_percentile < 0.2:
            regime = 'very_low_volatility'
        elif vol_percentile < 0.4:
            regime = 'low_volatility'
        elif vol_percentile < 0.6:
            regime = 'normal_volatility'
        elif vol_percentile < 0.8:
            regime = 'high_volatility'
        else:
            regime = 'very_high_volatility'
        
        return {
            'regime': regime,
            'percentile': vol_percentile,
            'current_vol': current_vol,
            'vol_trend': 'increasing' if vol_measures['realized_vol_10'].iloc[-5:].mean() > vol_measures['realized_vol_10'].iloc[-10:-5].mean() else 'decreasing'
        }
    
    def detect_trend_regime(self, data):
        """Detect trend regime"""
        # Multiple timeframe trend analysis
        trends = {}
        
        for period in [20, 50, 200]:
            sma = data['close'].rolling(period).mean()
            
            if len(data) >= period:
                current_price = data['close'].iloc[-1]
                sma_value = sma.iloc[-1]
                
                # Trend direction
                if current_price > sma_value * 1.02:
                    direction = 'strong_uptrend'
                elif current_price > sma_value:
                    direction = 'uptrend'
                elif current_price < sma_value * 0.98:
                    direction = 'strong_downtrend'
                elif current_price < sma_value:
                    direction = 'downtrend'
                else:
                    direction = 'sideways'
                
                # Trend strength
                distances = []
                for i in range(min(20, len(data) - period)):
                    distances.append((data['close'].iloc[-(i+1)] - sma.iloc[-(i+1)]) / sma.iloc[-(i+1)])
                
                trend_strength = np.mean(distances) if distances else 0
                
                trends[f'trend_{period}'] = {
                    'direction': direction,
                    'strength': abs(trend_strength),
                    'consistency': np.std(distances) if distances else 0
                }
        
        # Overall trend regime
        if all(t['direction'].endswith('uptrend') for t in trends.values()):
            overall_trend = 'strong_bull_market'
        elif all('uptrend' in t['direction'] for t in trends.values()):
            overall_trend = 'bull_market'
        elif all(t['direction'].endswith('downtrend') for t in trends.values()):
            overall_trend = 'strong_bear_market'
        elif all('downtrend' in t['direction'] for t in trends.values()):
            overall_trend = 'bear_market'
        else:
            overall_trend = 'mixed_market'
        
        return {
            'overall': overall_trend,
            'timeframes': trends
        }
    
    def calculate_garch_volatility(self, returns):
        """Calculate GARCH volatility"""
        try:
            from arch import arch_model
            
            # Remove NaN values
            returns_clean = returns.dropna()
            
            # Fit GARCH(1,1) model
            model = arch_model(returns_clean, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Get conditional volatility
            garch_vol = model_fit.conditional_volatility
            
            # Align with original index
            return pd.Series(index=returns.index).fillna(method='ffill').fillna(garch_vol.mean())
            
        except:
            # Fallback to simple volatility
            return returns.rolling(20).std()
    
    def combine_regimes(self, hmm_regime, vol_regime, trend_regime):
        """Combine different regime indicators"""
        # Create regime matrix
        regime_scores = {
            'risk_on': 0,
            'risk_off': 0,
            'neutral': 0
        }
        
        # HMM regime contribution
        if hmm_regime['current'] in [0, 1]:  # Assuming 0,1 are bull regimes
            regime_scores['risk_on'] += 0.4
        elif hmm_regime['current'] in [2, 3]:  # Assuming 2,3 are bear regimes
            regime_scores['risk_off'] += 0.4
        
        # Volatility regime contribution
        if vol_regime['regime'] in ['very_low_volatility', 'low_volatility']:
            regime_scores['risk_on'] += 0.3
        elif vol_regime['regime'] in ['high_volatility', 'very_high_volatility']:
            regime_scores['risk_off'] += 0.3
        else:
            regime_scores['neutral'] += 0.3
        
        # Trend regime contribution
        if 'bull' in trend_regime['overall']:
            regime_scores['risk_on'] += 0.3
        elif 'bear' in trend_regime['overall']:
            regime_scores['risk_off'] += 0.3
        else:
            regime_scores['neutral'] += 0.3
        
        # Determine overall regime
        overall_regime = max(regime_scores, key=regime_scores.get)
        
        return {
            'regime': overall_regime,
            'scores': regime_scores,
            'confidence': max(regime_scores.values()),
            'sub_regimes': {
                'hmm': hmm_regime,
                'volatility': vol_regime,
                'trend': trend_regime
            }
        }
    
    def get_regime_trading_rules(self, regime):
        """Get trading rules based on regime"""
        rules = {
            'risk_on': {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'preferred_strategies': ['trend_following', 'momentum'],
                'avoid_strategies': ['mean_reversion'],
                'max_positions': 5
            },
            'risk_off': {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.2,
                'preferred_strategies': ['mean_reversion', 'hedged'],
                'avoid_strategies': ['trend_following'],
                'max_positions': 2
            },
            'neutral': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.5,
                'preferred_strategies': ['balanced'],
                'avoid_strategies': [],
                'max_positions': 3
            }
        }
        
        return rules.get(regime['regime'], rules['neutral'])