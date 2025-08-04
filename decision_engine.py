import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

class DecisionEngine:
    """Advanced decision making engine for the AI agent"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
        # Decision models
        self.decision_models = {}
        
        # Decision history
        self.decision_history = []
        
        # Decision thresholds
        self.thresholds = {
            'min_confidence': 0.65,
            'max_correlation': 0.7,
            'max_risk_per_decision': 0.02
        }
    
    async def make_decisions(self, observations: Dict) -> List[Dict]:
        """Make trading decisions based on observations"""
        
        decisions = []
        
        # 1. Analyze market conditions
        market_analysis = await self.analyze_markets(observations)
        
        # 2. Generate trading signals
        signals = await self.generate_signals(market_analysis)
        
        # 3. Filter signals
        filtered_signals = self.filter_signals(signals)
        
        # 4. Risk assessment
        risk_approved_signals = await self.assess_risk(filtered_signals)
        
        # 5. Create final decisions
        for signal in risk_approved_signals:
            decision = self.create_decision(signal, market_analysis)
            decisions.append(decision)
        
        # 6. Record decisions
        self.record_decisions(decisions)
        
        return decisions
    
    async def analyze_markets(self, observations: Dict) -> Dict:
        """Comprehensive market analysis"""
        
        analysis = {
            'timestamp': datetime.now(),
            'market_conditions': {},
            'correlations': {},
            'opportunities': [],
            'risks': []
        }
        
        # Analyze each market
        for symbol, data in observations.get('market_data', {}).items():
            symbol_analysis = await self.analyze_symbol(symbol, data)
            analysis['market_conditions'][symbol] = symbol_analysis
            
            # Check for opportunities
            if symbol_analysis.get('signal_strength', 0) > 0.7:
                analysis['opportunities'].append({
                    'symbol': symbol,
                    'type': symbol_analysis['signal_type'],
                    'strength': symbol_analysis['signal_strength']
                })
        
        # Calculate correlations
        analysis['correlations'] = self.calculate_correlations(observations)
        
        # Identify risks
        analysis['risks'] = self.identify_risks(analysis)
        
        return analysis
    
    async def analyze_symbol(self, symbol: str, data: Dict) -> Dict:
        """Analyze individual symbol"""
        
        analysis = {
            'symbol': symbol,
            'trend': 'neutral',
            'momentum': 0,
            'volatility': 0,
            'signal_type': None,
            'signal_strength': 0
        }
        
        # Get hourly data
        df = data.get('H1', data.get('1h'))
        if df is None or len(df) < 50:
            return analysis
        
        # Calculate indicators
        # Trend
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            analysis['trend'] = 'bullish'
        elif sma_20.iloc[-1] < sma_50.iloc[-1]:
            analysis['trend'] = 'bearish'
        
        # Momentum
        rsi = self.calculate_rsi(df['close'])
        analysis['momentum'] = (rsi.iloc[-1] - 50) / 50  # Normalize to -1 to 1
        
        # Volatility
        analysis['volatility'] = df['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Generate signal
        signal = self.generate_symbol_signal(analysis, df)
        analysis.update(signal)
        
        return analysis
    
    def generate_symbol_signal(self, analysis: Dict, df: pd.DataFrame) -> Dict:
        """Generate trading signal for symbol"""
        
        signal = {
            'signal_type': None,
            'signal_strength': 0
        }
        
        # Trend following signal
        if analysis['trend'] == 'bullish' and analysis['momentum'] > 0.2:
            signal['signal_type'] = 'buy'
            signal['signal_strength'] = min(1.0, analysis['momentum'] + 0.5)
        elif analysis['trend'] == 'bearish' and analysis['momentum'] < -0.2:
            signal['signal_type'] = 'sell'
            signal['signal_strength'] = min(1.0, abs(analysis['momentum']) + 0.5)
        
        # Mean reversion signal (if no trend signal)
        if signal['signal_type'] is None:
            rsi = self.calculate_rsi(df['close'])
            if rsi.iloc[-1] < 30:
                signal['signal_type'] = 'buy'
                signal['signal_strength'] = (30 - rsi.iloc[-1]) / 30
            elif rsi.iloc[-1] > 70:
                signal['signal_type'] = 'sell'
                signal['signal_strength'] = (rsi.iloc[-1] - 70) / 30
        
        return signal
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_correlations(self, observations: Dict) -> Dict:
        """Calculate correlations between symbols"""
        
        correlations = {}
        
        symbols = list(observations.get('market_data', {}).keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Get price data
                data1 = observations['market_data'][symbol1].get('H1', 
                        observations['market_data'][symbol1].get('1h'))
                data2 = observations['market_data'][symbol2].get('H1',
                        observations['market_data'][symbol2].get('1h'))
                
                if data1 is not None and data2 is not None:
                    # Calculate correlation
                    min_len = min(len(data1), len(data2))
                    if min_len > 20:
                        corr = data1['close'].iloc[-min_len:].corr(
                               data2['close'].iloc[-min_len:])
                        correlations[f"{symbol1}_{symbol2}"] = corr
        
        return correlations
    
    def identify_risks(self, analysis: Dict) -> List[Dict]:
        """Identify potential risks"""
        
        risks = []
        
        # High correlation risk
        for pair, corr in analysis['correlations'].items():
            if abs(corr) > self.thresholds['max_correlation']:
                risks.append({
                    'type': 'high_correlation',
                    'symbols': pair.split('_'),
                    'value': corr
                })
        
        # High volatility risk
        for symbol, condition in analysis['market_conditions'].items():
            if condition['volatility'] > 0.03:
                risks.append({
                    'type': 'high_volatility',
                    'symbol': symbol,
                    'value': condition['volatility']
                })
        
        return risks
    
    async def generate_signals(self, analysis: Dict) -> List[Dict]:
        """Generate trading signals from analysis"""
        
        signals = []
        
        for opportunity in analysis['opportunities']:
            signal = {
                'symbol': opportunity['symbol'],
                'action': opportunity['type'],
                'strength': opportunity['strength'],
                'analysis': analysis['market_conditions'][opportunity['symbol']],
                'timestamp': datetime.now()
            }
            
            # Add ML predictions if available
            if hasattr(self.system, 'models') and self.system.models:
                prediction = await self.get_ml_prediction(opportunity['symbol'])
                signal['ml_confidence'] = prediction
            else:
                signal['ml_confidence'] = 1.0  # Default confidence
            
            signals.append(signal)
        
        return signals
    
    async def get_ml_prediction(self, symbol: str) -> float:
        """Get ML model prediction for symbol"""
        
        try:
            # Get latest features
            timeframe = 'H1' if self.system.active_market == 'forex' else '1h'
            data = await self.system.connector.get_historical_data(symbol, timeframe, 500)
            
            if data is None or len(data) < 100:
                return 0.5
            
            # Create features
            featured_data = self.system.feature_engineer.create_all_features(data)
            latest_features = featured_data[self.system.feature_engineer.feature_names].iloc[-1:]
            
            # Get predictions
            predictions = []
            for model in self.system.models.values():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(latest_features)[0]
                    predictions.append(np.max(pred))
            
            return np.mean(predictions) if predictions else 0.5
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction: {e}")
            return 0.5
    
    def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """Filter signals based on criteria"""
        
        filtered = []
        
        for signal in signals:
            # Check minimum confidence
            confidence = signal.get('strength', 0) * signal.get('ml_confidence', 1)
            
            if confidence >= self.thresholds['min_confidence']:
                signal['final_confidence'] = confidence
                filtered.append(signal)
        
        # Sort by confidence
        filtered.sort(key=lambda x: x['final_confidence'], reverse=True)
        
        # Limit number of signals
        return filtered[:5]
    
    async def assess_risk(self, signals: List[Dict]) -> List[Dict]:
        """Assess risk for each signal"""
        
        approved_signals = []
        
        # Get current positions
        positions = []
        account_info = {'balance': 10000}  # Default
        
        # Try to get real positions and account info
        try:
            if hasattr(self.system, 'connector'):
                positions = await self.system.connector.get_open_positions()
                account_info = await self.system.connector.get_account_info()
        except:
            pass
        
        current_exposure = sum(p.get('volume', 0) for p in positions)
        max_exposure = account_info.get('balance', 10000) * 0.1
        
        for signal in signals:
            # Check if we already have position in this symbol
            symbol_positions = [p for p in positions if p['symbol'] == signal['symbol']]
            
            if symbol_positions:
                continue  # Skip if already have position
            
            # Check total exposure
            if current_exposure < max_exposure:
                approved_signals.append(signal)
                current_exposure += account_info.get('balance', 10000) * self.thresholds['max_risk_per_decision']
        
        return approved_signals
    
    def create_decision(self, signal: Dict, analysis: Dict) -> Dict:
        """Create final decision from signal"""
        
        decision = {
            'type': 'trade',
            'symbol': signal['symbol'],
            'direction': signal['action'],
            'confidence': signal['final_confidence'],
            'stop_loss_pct': self.calculate_stop_loss(signal, analysis),
            'take_profit_pct': self.calculate_take_profit(signal, analysis),
            'analysis': signal['analysis'],
            'timestamp': datetime.now()
        }
        
        return decision
    
    def calculate_stop_loss(self, signal: Dict, analysis: Dict) -> float:
        """Calculate appropriate stop loss"""
        
        base_stop = 0.02  # 2% base stop loss
        
        # Get volatility from the analysis
        volatility = signal['analysis'].get('volatility', 0)
        
        # Adjust the base stop-loss based on volatility
        if volatility > 0.02:  # High volatility
            base_stop *= 1.5
        elif volatility < 0.01:  # Low volatility
            base_stop *= 0.75
        
        # Ensure the stop-loss is within reasonable bounds
        base_stop = max(0.01, min(0.05, base_stop))  # Clamp between 1% and 5%
        
        return base_stop
    
    def calculate_take_profit(self, signal: Dict, analysis: Dict) -> float:
        """Calculate appropriate take profit"""
        
        stop_loss = self.calculate_stop_loss(signal, analysis)
        
        # Risk-reward ratio based on confidence
        confidence = signal['final_confidence']
        
        if confidence > 0.8:
            risk_reward = 3.0
        elif confidence > 0.7:
            risk_reward = 2.0
        else:
            risk_reward = 1.5
        
        return stop_loss * risk_reward
    
    def record_decisions(self, decisions: List[Dict]):
        """Record decisions for analysis"""
        
        for decision in decisions:
            self.decision_history.append(decision)
        
        # Keep history size manageable
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]