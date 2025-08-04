import numpy as np
import pandas as pd
import ta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
        
    def identify_chart_patterns(self, df):
        """Identify complex chart patterns"""
        patterns = pd.DataFrame(index=df.index)
        
        # Head and Shoulders
        patterns['head_shoulders'] = self._detect_head_shoulders(df)
        
        # Double Top/Bottom
        patterns['double_top'] = self._detect_double_top(df)
        patterns['double_bottom'] = self._detect_double_bottom(df)
        
        # Triangle patterns
        patterns['ascending_triangle'] = self._detect_ascending_triangle(df)
        patterns['descending_triangle'] = self._detect_descending_triangle(df)
        patterns['symmetrical_triangle'] = self._detect_symmetrical_triangle(df)
        
        # Flag and Pennant
        patterns['bull_flag'] = self._detect_bull_flag(df)
        patterns['bear_flag'] = self._detect_bear_flag(df)
        
        # Cup and Handle
        patterns['cup_handle'] = self._detect_cup_and_handle(df)
        
        return patterns
    
    def _detect_head_shoulders(self, df, window=20):
        """Detect head and shoulders pattern"""
        highs = df['high'].rolling(window=window).max()
        
        # Find local maxima
        peaks = argrelextrema(highs.values, np.greater, order=5)[0]
        
        pattern_signals = np.zeros(len(df))
        
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                left_shoulder = highs.iloc[peaks[i]]
                head = highs.iloc[peaks[i + 1]]
                right_shoulder = highs.iloc[peaks[i + 2]]
                
                # Check if middle peak is highest (head)
                if head > left_shoulder and head > right_shoulder:
                    # Check if shoulders are approximately equal
                    if abs(left_shoulder - right_shoulder) / left_shoulder < 0.03:
                        pattern_signals[peaks[i + 2]] = -1  # Bearish signal
        
        return pattern_signals
    
    def _detect_double_top(self, df, window=20, tolerance=0.02):
        """Detect double top pattern"""
        highs = df['high'].rolling(window=window).max()
        peaks = argrelextrema(highs.values, np.greater, order=5)[0]
        
        pattern_signals = np.zeros(len(df))
        
        for i in range(len(peaks) - 1):
            peak1 = highs.iloc[peaks[i]]
            peak2 = highs.iloc[peaks[i + 1]]
            
            # Check if peaks are approximately equal
            if abs(peak1 - peak2) / peak1 < tolerance:
                # Check if there's a valley between peaks
                valley_start = peaks[i]
                valley_end = peaks[i + 1]
                valley = df['low'].iloc[valley_start:valley_end].min()
                
                if valley < peak1 * 0.95:  # At least 5% retracement
                    pattern_signals[peaks[i + 1]] = -1  # Bearish signal
        
        return pattern_signals
    
    def _detect_double_bottom(self, df, window=20, tolerance=0.02):
        """Detect double bottom pattern"""
        lows = df['low'].rolling(window=window).min()
        troughs = argrelextrema(lows.values, np.less, order=5)[0]
        
        pattern_signals = np.zeros(len(df))
        
        for i in range(len(troughs) - 1):
            trough1 = lows.iloc[troughs[i]]
            trough2 = lows.iloc[troughs[i + 1]]
            
            # Check if troughs are approximately equal
            if abs(trough1 - trough2) / trough1 < tolerance:
                # Check if there's a peak between troughs
                peak_start = troughs[i]
                peak_end = troughs[i + 1]
                peak = df['high'].iloc[peak_start:peak_end].max()
                
                if peak > trough1 * 1.05:  # At least 5% bounce
                    pattern_signals[troughs[i + 1]] = 1  # Bullish signal
        
        return pattern_signals
    
    def _detect_ascending_triangle(self, df, window=30):
        """Detect ascending triangle pattern"""
        pattern_signals = np.zeros(len(df))
        
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Check for horizontal resistance
            resistance_level = segment['high'].max()
            resistance_touches = np.sum(abs(segment['high'] - resistance_level) / resistance_level < 0.01)
            
            if resistance_touches >= 2:
                # Check for ascending support
                lows = segment['low'].values
                x = np.arange(len(lows)).reshape(-1, 1)
                
                reg = LinearRegression().fit(x, lows)
                slope = reg.coef_[0]
                
                if slope > 0:  # Ascending trendline
                    r_squared = reg.score(x, lows)
                    if r_squared > 0.8:  # Good fit
                        pattern_signals[i] = 1  # Bullish signal
        
        return pattern_signals
    
    def _detect_descending_triangle(self, df, window=30):
        """Detect descending triangle pattern"""
        pattern_signals = np.zeros(len(df))
        
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Check for horizontal support
            support_level = segment['low'].min()
            support_touches = np.sum(abs(segment['low'] - support_level) / support_level < 0.01)
            
            if support_touches >= 2:
                # Check for descending resistance
                highs = segment['high'].values
                x = np.arange(len(highs)).reshape(-1, 1)
                
                reg = LinearRegression().fit(x, highs)
                slope = reg.coef_[0]
                
                if slope < 0:  # Descending trendline
                    r_squared = reg.score(x, highs)
                    if r_squared > 0.8:  # Good fit
                        pattern_signals[i] = -1  # Bearish signal
        
        return pattern_signals
    
    def _detect_symmetrical_triangle(self, df, window=30):
        """Detect symmetrical triangle pattern"""
        pattern_signals = np.zeros(len(df))
        
        for i in range(window, len(df)):
            segment = df.iloc[i-window:i]
            
            # Fit trendlines to highs and lows
            x = np.arange(len(segment)).reshape(-1, 1)
            
            high_reg = LinearRegression().fit(x, segment['high'].values)
            low_reg = LinearRegression().fit(x, segment['low'].values)
            
            high_slope = high_reg.coef_[0]
            low_slope = low_reg.coef_[0]
            
            # Check for converging trendlines
            if high_slope < 0 and low_slope > 0:  # Converging
                high_r2 = high_reg.score(x, segment['high'].values)
                low_r2 = low_reg.score(x, segment['low'].values)
                
                if high_r2 > 0.8 and low_r2 > 0.8:  # Good fit
                    # Determine breakout direction based on prior trend
                    prior_trend = df['close'].iloc[i-window*2:i-window].mean()
                    current_price = df['close'].iloc[i]
                    
                    if current_price > prior_trend:
                        pattern_signals[i] = 1  # Bullish
                    else:
                        pattern_signals[i] = -1  # Bearish
        
        return pattern_signals
    
    def _detect_bull_flag(self, df, window=20):
        """Detect bull flag pattern"""
        pattern_signals = np.zeros(len(df))
        
        for i in range(window*2, len(df)):
            # Look for strong upward move (pole)
            pole_start = i - window*2
            pole_end = i - window
            pole_return = (df['close'].iloc[pole_end] - df['close'].iloc[pole_start]) / df['close'].iloc[pole_start]
            
            if pole_return > 0.1:  # 10% move up
                # Look for consolidation (flag)
                flag_segment = df.iloc[pole_end:i]
                flag_high = flag_segment['high'].max()
                flag_low = flag_segment['low'].min()
                flag_range = (flag_high - flag_low) / flag_low
                
                if flag_range < 0.05:  # Tight consolidation
                    # Check for slight downward bias
                    x = np.arange(len(flag_segment)).reshape(-1, 1)
                    reg = LinearRegression().fit(x, flag_segment['close'].values)
                    
                    if -0.02 < reg.coef_[0] < 0:  # Slight downward slope
                        pattern_signals[i] = 1  # Bullish signal
        
        return pattern_signals
    
    def _detect_bear_flag(self, df, window=20):
        """Detect bear flag pattern"""
        pattern_signals = np.zeros(len(df))
        
        for i in range(window*2, len(df)):
            # Look for strong downward move (pole)
            pole_start = i - window*2
            pole_end = i - window
            pole_return = (df['close'].iloc[pole_end] - df['close'].iloc[pole_start]) / df['close'].iloc[pole_start]
            
            if pole_return < -0.1:  # 10% move down
                # Look for consolidation (flag)
                flag_segment = df.iloc[pole_end:i]
                flag_high = flag_segment['high'].max()
                flag_low = flag_segment['low'].min()
                flag_range = (flag_high - flag_low) / flag_low
                
                if flag_range < 0.05:  # Tight consolidation
                    # Check for slight upward bias
                    x = np.arange(len(flag_segment)).reshape(-1, 1)
                    reg = LinearRegression().fit(x, flag_segment['close'].values)
                    
                    if 0 < reg.coef_[0] < 0.02:  # Slight upward slope
                        pattern_signals[i] = -1  # Bearish signal
        
        return pattern_signals
    
    def _detect_cup_and_handle(self, df, window=50):
        """Detect cup and handle pattern"""
        pattern_signals = np.zeros(len(df))
        
        for i in range(window + 20, len(df)):
            # Look for cup formation
            cup_segment = df.iloc[i-window:i-10]
            cup_start_price = cup_segment['close'].iloc[0]
            cup_end_price = cup_segment['close'].iloc[-1]
            
            # Check if prices are similar at start and end
            if abs(cup_start_price - cup_end_price) / cup_start_price < 0.05:
                # Check for U-shape
                cup_min = cup_segment['low'].min()
                cup_depth = (cup_start_price - cup_min) / cup_start_price
                
                if 0.1 < cup_depth < 0.3:  # Reasonable cup depth
                    # Look for handle
                    handle_segment = df.iloc[i-10:i]
                    handle_high = handle_segment['high'].max()
                    handle_low = handle_segment['low'].min()
                    handle_range = (handle_high - handle_low) / handle_low
                    
                    if handle_range < 0.1:  # Small consolidation
                        # Check if handle is in upper half of cup
                        if handle_low > (cup_min + cup_start_price) / 2:
                            pattern_signals[i] = 1  # Bullish signal
        
        return pattern_signals
    
    def calculate_market_breadth(self, df_list):
        """Calculate market breadth indicators"""
        breadth_indicators = {}
        
        # Advance/Decline Line
        advances = sum(1 for df in df_list if df['close'].iloc[-1] > df['close'].iloc[-2])
        declines = sum(1 for df in df_list if df['close'].iloc[-1] < df['close'].iloc[-2])
        breadth_indicators['advance_decline_ratio'] = advances / (declines + 1)
        
        # New Highs/Lows
        new_highs = sum(1 for df in df_list if df['high'].iloc[-1] == df['high'].rolling(52).max().iloc[-1])
        new_lows = sum(1 for df in df_list if df['low'].iloc[-1] == df['low'].rolling(52).min().iloc[-1])
        breadth_indicators['new_highs_lows_ratio'] = new_highs / (new_lows + 1)
        
        # McClellan Oscillator (simplified)
        breadth_indicators['mcclellan_oscillator'] = (advances - declines) / len(df_list)
        
        return breadth_indicators
    
    def calculate_market_internals(self, df):
        """Calculate market internal indicators"""
        internals = {}
        
        # TICK indicator (simplified)
        upticks = (df['close'] > df['close'].shift(1)).sum()
        downticks = (df['close'] < df['close'].shift(1)).sum()
        internals['tick'] = upticks - downticks
        
                # TRIN (Arms Index)
        if 'volume' in df.columns:
            up_volume = df.loc[df['close'] > df['close'].shift(1), 'volume'].sum()
            down_volume = df.loc[df['close'] < df['close'].shift(1), 'volume'].sum()
            
            internals['trin'] = (upticks / (downticks + 1)) / (up_volume / (down_volume + 1))
        
        # Put/Call Ratio (would need options data)
        internals['put_call_ratio'] = 1.0  # Placeholder
        
        return internals