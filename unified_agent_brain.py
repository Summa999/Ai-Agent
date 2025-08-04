import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
from abc import ABC, abstractmethod

class MarketPersonality:
    """AI Agent personality traits for different markets"""
    
    def __init__(self, market_type: str):
        self.market_type = market_type
        
        if market_type == 'forex':
            self.traits = {
                'patience': 0.8,      # Forex needs more patience
                'aggression': 0.3,    # Less aggressive in forex
                'precision': 0.9,     # High precision needed
                'adaptability': 0.7,  # Moderate adaptability
                'risk_tolerance': 0.4 # Lower risk in forex
            }
        elif market_type == 'crypto':
            self.traits = {
                'patience': 0.5,      # Less patience in volatile crypto
                'aggression': 0.6,    # More aggressive
                'precision': 0.7,     # Still need precision
                'adaptability': 0.9,  # High adaptability for 24/7 market
                'risk_tolerance': 0.6 # Higher risk tolerance
            }
        else:
            self.traits = {
                'patience': 0.6,
                'aggression': 0.5,
                'precision': 0.8,
                'adaptability': 0.8,
                'risk_tolerance': 0.5
            }

class UnifiedAgentBrain:
    """Unified AI Agent Brain that works with all markets"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
        # Multi-market state
        self.market_states = {
            'forex': {
                'active': False,
                'personality': MarketPersonality('forex'),
                'strategies': [],
                'memory': [],
                'performance': {}
            },
            'crypto': {
                'active': False,
                'personality': MarketPersonality('crypto'),
                'strategies': [],
                'memory': [],
                'performance': {}
            }
        }
        
        # Unified memory and learning
        self.global_memory = {
            'successful_patterns': {},
            'market_correlations': {},
            'cross_market_insights': {},
            'strategy_performance': {}
        }
        
        # Agent state
        self.thinking_mode = 'analytical'  # analytical, intuitive, hybrid
        self.confidence_level = 0.5
        self.learning_rate = 0.01
        
        # Initialize sub-brains for specialized thinking
        self.forex_brain = ForexSpecialistBrain(self)
        self.crypto_brain = CryptoSpecialistBrain(self)
        self.arbitrage_brain = ArbitrageBrain(self)
        
    async def think(self, observations: Dict) -> List[Dict]:
        """Main thinking process for all markets"""
        
        decisions = []
        
        # 1. Process observations for each active market
        market_insights = await self.process_observations(observations)
        
        # 2. Cross-market analysis
        cross_market_opportunities = await self.analyze_cross_markets(market_insights)
        
        # 3. Generate decisions for each market
        for market_type in ['forex', 'crypto']:
            if self.market_states[market_type]['active']:
                market_decisions = await self.think_for_market(
                    market_type, 
                    market_insights.get(market_type, {}),
                    cross_market_opportunities
                )
                decisions.extend(market_decisions)
        
        # 4. Arbitrage opportunities
        arbitrage_decisions = await self.arbitrage_brain.find_opportunities(
            market_insights
        )
        decisions.extend(arbitrage_decisions)
        
        # 5. Portfolio optimization across markets
        optimized_decisions = await self.optimize_portfolio_decisions(decisions)
        
        return optimized_decisions
    
    async def process_observations(self, observations: Dict) -> Dict:
        """Process observations from all markets"""
        
        insights = {}
        
        # Process each market's observations
        for market_name, market_data in observations.get('markets', {}).items():
            if market_name == 'forex':
                insights['forex'] = await self.forex_brain.analyze(market_data)
                self.market_states['forex']['active'] = True
            elif market_name == 'crypto':
                insights['crypto'] = await self.crypto_brain.analyze(market_data)
                self.market_states['crypto']['active'] = True
        
        return insights
    
    async def think_for_market(self, market_type: str, market_data: Dict, 
                               cross_market_data: Dict) -> List[Dict]:
        """Generate decisions for specific market"""
        
        decisions = []
        personality = self.market_states[market_type]['personality']
        
        # 1. Identify opportunities
        opportunities = await self.identify_opportunities(market_type, market_data)
        
        # 2. Apply market-specific strategies
        for opportunity in opportunities:
            # Evaluate with market personality
            score = self.evaluate_opportunity(opportunity, personality)
            
            if score > 0.6:
                decision = await self.create_market_decision(
                    market_type,
                    opportunity,
                    score,
                    cross_market_data
                )
                decisions.append(decision)
        
        # 3. Risk management decisions
        risk_decisions = await self.assess_market_risk(market_type, market_data)
        decisions.extend(risk_decisions)
        
        return decisions
    
    async def identify_opportunities(self, market_type: str, market_data: Dict) -> List[Dict]:
        """Identify trading opportunities for specific market"""
        
        opportunities = []
        
        if market_type == 'forex':
            # Forex-specific opportunities
            opportunities.extend(
                await self.forex_brain.find_opportunities(market_data)
            )
        elif market_type == 'crypto':
            # Crypto-specific opportunities
            opportunities.extend(
                await self.crypto_brain.find_opportunities(market_data)
            )
        
        # Common opportunities (work for both markets)
        opportunities.extend(
            self.find_common_opportunities(market_data)
        )
        
        return opportunities
    
    def find_common_opportunities(self, market_data: Dict) -> List[Dict]:
        """Find opportunities that work in any market"""
        
        opportunities = []
        
        # Trend following opportunities
        for symbol, data in market_data.get('symbols', {}).items():
            if self.detect_strong_trend(data):
                opportunities.append({
                    'type': 'trend_following',
                    'symbol': symbol,
                    'direction': data['trend_direction'],
                    'strength': data['trend_strength'],
                    'strategy': 'adaptive_trend'
                })
            
            # Mean reversion opportunities
            if self.detect_oversold_overbought(data):
                opportunities.append({
                    'type': 'mean_reversion',
                    'symbol': symbol,
                    'direction': data['reversion_direction'],
                    'strength': data['reversion_strength'],
                    'strategy': 'smart_reversion'
                })
        
        return opportunities
    
    def detect_strong_trend(self, data: Dict) -> bool:
        """Detect strong trend in any market"""
        
        if 'indicators' not in data:
            return False
        
        indicators = data['indicators']
        
        # Multiple confirmations needed
        confirmations = 0
        
        # ADX above 25
        if indicators.get('adx', 0) > 25:
            confirmations += 1
        
        # Price above/below moving averages
        if indicators.get('price_ma_alignment', False):
            confirmations += 1
        
        # MACD alignment
        if indicators.get('macd_aligned', False):
            confirmations += 1
        
        # Update data with trend info
        if confirmations >= 2:
            data['trend_direction'] = 'buy' if indicators.get('trend_bias', 0) > 0 else 'sell'
            data['trend_strength'] = min(1.0, confirmations / 3)
            return True
        
        return False
    
    def detect_oversold_overbought(self, data: Dict) -> bool:
        """Detect oversold/overbought conditions"""
        
        if 'indicators' not in data:
            return False
        
        indicators = data['indicators']
        rsi = indicators.get('rsi', 50)
        
        # Extreme RSI with other confirmations
        if rsi < 30:
            data['reversion_direction'] = 'buy'
            data['reversion_strength'] = (30 - rsi) / 30
            return True
        elif rsi > 70:
            data['reversion_direction'] = 'sell'
            data['reversion_strength'] = (rsi - 70) / 30
            return True
        
        return False
    
    def evaluate_opportunity(self, opportunity: Dict, personality: MarketPersonality) -> float:
        """Evaluate opportunity based on market personality"""
        
        base_score = opportunity.get('strength', 0.5)
        
        # Adjust based on personality traits
        if opportunity['type'] == 'trend_following':
            # Patient personalities prefer trends
            base_score *= (1 + personality.traits['patience'] * 0.3)
        elif opportunity['type'] == 'mean_reversion':
            # Aggressive personalities prefer quick reversions
            base_score *= (1 + personality.traits['aggression'] * 0.3)
        
        # Risk adjustment
        risk_factor = opportunity.get('risk_level', 0.5)
        if risk_factor > personality.traits['risk_tolerance']:
            base_score *= 0.7  # Reduce score for high risk
        
        # Precision requirement
        if opportunity.get('requires_precision', False):
            base_score *= personality.traits['precision']
        
        return min(1.0, max(0.0, base_score))
    
    async def create_market_decision(self, market_type: str, opportunity: Dict,
                                    score: float, cross_market_data: Dict) -> Dict:
        """Create a trading decision for specific market"""
        
        decision = {
            'type': 'trade',
            'market': market_type,
            'symbol': opportunity['symbol'],
            'direction': opportunity['direction'],
            'confidence': score,
            'strategy': opportunity['strategy'],
            'timestamp': datetime.now()
        }
        
        # Market-specific parameters
        if market_type == 'forex':
            decision.update(self.forex_brain.get_trade_parameters(opportunity, score))
        elif market_type == 'crypto':
            decision.update(self.crypto_brain.get_trade_parameters(opportunity, score))
        
        # Cross-market adjustments
        if cross_market_data.get('high_correlation_alert'):
            decision['position_size_multiplier'] = 0.7  # Reduce size
        
        return decision
    
    async def analyze_cross_markets(self, market_insights: Dict) -> Dict:
        """Analyze relationships between markets"""
        
        cross_market_data = {
            'correlations': {},
            'divergences': [],
            'arbitrage_opportunities': [],
            'risk_indicators': {}
        }
        
        # Only analyze if both markets have data
        if 'forex' in market_insights and 'crypto' in market_insights:
            # Check correlations (e.g., EURUSD vs BTC)
            correlations = self.calculate_market_correlations(
                market_insights['forex'],
                market_insights['crypto']
            )
            cross_market_data['correlations'] = correlations
            
            # Check for divergences
            divergences = self.find_market_divergences(
                market_insights['forex'],
                market_insights['crypto']
            )
            cross_market_data['divergences'] = divergences
            
            # Global risk indicators
            cross_market_data['risk_indicators'] = {
                'global_volatility': self.calculate_global_volatility(market_insights),
                'risk_on_off': self.determine_risk_sentiment(market_insights)
            }
        
        return cross_market_data
    
    def calculate_market_correlations(self, forex_data: Dict, crypto_data: Dict) -> Dict:
        """Calculate correlations between forex and crypto markets"""
        
        correlations = {}
        
        # Example: EURUSD often inversely correlated with BTC/USD
        if 'EURUSD' in forex_data.get('symbols', {}) and \
           'BTC/USDT' in crypto_data.get('symbols', {}):
            
            # Simplified correlation calculation
            eur_trend = forex_data['symbols']['EURUSD'].get('trend_strength', 0)
            btc_trend = crypto_data['symbols']['BTC/USDT'].get('trend_strength', 0)
            
            correlation = -0.3  # Base inverse correlation
            if eur_trend * btc_trend < 0:  # Opposite directions
                correlation = -0.6
            
            correlations['EURUSD_BTC'] = correlation
        
        return correlations
    
    def find_market_divergences(self, forex_data: Dict, crypto_data: Dict) -> List[Dict]:
        """Find divergences between markets"""
        divergences = []
        # Implementation for finding divergences
        return divergences
    
    def calculate_global_volatility(self, market_insights: Dict) -> float:
        """Calculate overall market volatility"""
        volatilities = []
        
        for market_data in market_insights.values():
            for symbol_data in market_data.get('symbols', {}).values():
                if 'volatility' in symbol_data:
                    volatilities.append(symbol_data['volatility'])
        
        return np.mean(volatilities) if volatilities else 0.0
    
    def determine_risk_sentiment(self, market_insights: Dict) -> str:
        """Determine overall risk sentiment"""
        # Simplified implementation
        global_vol = self.calculate_global_volatility(market_insights)
        
        if global_vol > 0.03:
            return 'risk_off'
        elif global_vol < 0.01:
            return 'risk_on'
        else:
            return 'neutral'
    
    async def optimize_portfolio_decisions(self, decisions: List[Dict]) -> List[Dict]:
        """Optimize decisions across entire portfolio"""
        
        if not decisions:
            return []
        
        # Get current portfolio state
        portfolio_state = await self.get_portfolio_state()
        
        # Risk budget allocation
        total_risk_budget = 0.06  # 6% total portfolio risk
        used_risk = portfolio_state.get('current_risk', 0)
        available_risk = total_risk_budget - used_risk
        
        # Sort decisions by confidence
        decisions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Allocate risk budget
        optimized_decisions = []
        allocated_risk = 0
        
        for decision in decisions:
            if decision['type'] != 'trade':
                optimized_decisions.append(decision)
                continue
            
            # Calculate risk for this decision
            decision_risk = self.calculate_decision_risk(decision)
            
            if allocated_risk + decision_risk <= available_risk:
                # Adjust position size based on portfolio
                decision = self.adjust_for_portfolio(decision, portfolio_state)
                optimized_decisions.append(decision)
                allocated_risk += decision_risk
            else:
                # Skip or reduce position
                if allocated_risk < available_risk * 0.5:
                    # We have room for a smaller position
                    decision['position_size_multiplier'] = 0.5
                    optimized_decisions.append(decision)
                    break
        
        return optimized_decisions
    
    async def get_portfolio_state(self) -> Dict:
        """Get current portfolio state across all markets"""
        
        state = {
            'total_positions': 0,
            'current_risk': 0,
            'market_exposure': {
                'forex': 0,
                'crypto': 0
            },
            'performance': {
                'daily_pnl': 0,
                'weekly_pnl': 0
            }
        }
        
        # This would connect to actual portfolio data
        # Simplified for example
        
        return state
    
    def calculate_decision_risk(self, decision: Dict) -> float:
        """Calculate risk for a decision"""
        
        base_risk = 0.01  # 1% base risk
        
        # Adjust based on market
        if decision['market'] == 'crypto':
            base_risk *= 1.5  # Higher risk in crypto
        
        # Adjust based on confidence
        confidence_multiplier = 2 - decision.get('confidence', 0.5)
        base_risk *= confidence_multiplier
        
        return base_risk
    
    def adjust_for_portfolio(self, decision: Dict, portfolio_state: Dict) -> Dict:
        """Adjust decision based on portfolio state"""
        
        # Reduce position if already heavily exposed to this market
        market_exposure = portfolio_state['market_exposure'].get(decision['market'], 0)
        
        if market_exposure > 0.4:  # 40% in one market
            decision['position_size_multiplier'] = 0.7
        
        # Increase position for well-performing strategies
        strategy_performance = self.global_memory['strategy_performance'].get(
            decision['strategy'], {}
        )
        
        if strategy_performance.get('win_rate', 0.5) > 0.6:
            current_multiplier = decision.get('position_size_multiplier', 1.0)
            decision['position_size_multiplier'] = current_multiplier * 1.2
        
        return decision
    
    async def assess_market_risk(self, market_type: str, market_data: Dict) -> List[Dict]:
        """Assess and generate risk management decisions"""
        decisions = []
        
        # Check for high volatility
        avg_volatility = np.mean([
            data.get('volatility', 0) 
            for data in market_data.get('symbols', {}).values()
        ])
        
        if avg_volatility > 0.03:
            decisions.append({
                'type': 'risk_adjustment',
                'market': market_type,
                'action': 'reduce_exposure',
                'reason': 'high_volatility',
                'parameters': {
                    'position_size_multiplier': 0.7,
                    'max_positions': 2
                }
            })
        
        return decisions
    
    async def learn_from_results(self, trade_results: List[Dict]):
        """Learn from trading results across all markets"""
        
        for result in trade_results:
            market = result.get('market')
            
            # Update market-specific memory
            if market in self.market_states:
                self.market_states[market]['memory'].append(result)
                
                # Keep memory size manageable
                if len(self.market_states[market]['memory']) > 1000:
                    self.market_states[market]['memory'] = \
                        self.market_states[market]['memory'][-1000:]
            
            # Update global patterns
            self.update_global_patterns(result)
            
            # Update strategy performance
            strategy = result.get('strategy')
            if strategy:
                if strategy not in self.global_memory['strategy_performance']:
                    self.global_memory['strategy_performance'][strategy] = {
                        'trades': 0,
                        'wins': 0,
                        'total_pnl': 0
                    }
                
                perf = self.global_memory['strategy_performance'][strategy]
                perf['trades'] += 1
                if result.get('profit', 0) > 0:
                    perf['wins'] += 1
                perf['total_pnl'] += result.get('profit', 0)
                perf['win_rate'] = perf['wins'] / perf['trades']
    
    def update_global_patterns(self, result: Dict):
        """Update global trading patterns"""
        
        if result.get('profit', 0) > 0:
            # Successful trade - extract pattern
            pattern_key = f"{result['market']}_{result['strategy']}_{result.get('market_condition', 'normal')}"
            
            if pattern_key not in self.global_memory['successful_patterns']:
                self.global_memory['successful_patterns'][pattern_key] = {
                    'count': 0,
                    'avg_profit': 0,
                    'conditions': []
                }
            
            pattern = self.global_memory['successful_patterns'][pattern_key]
            pattern['count'] += 1
            pattern['avg_profit'] = (
                (pattern['avg_profit'] * (pattern['count'] - 1) + result['profit']) /
                pattern['count']
            )
            pattern['conditions'].append(result.get('entry_conditions', {}))
    
    def get_thinking_interval(self) -> int:
        """Dynamic thinking interval based on market conditions"""
        
        # Base interval
        interval = 60  # 60 seconds
        
        # Adjust based on volatility
        global_vol = 0
        for market_state in self.market_states.values():
            if market_state['active']:
                # Get average volatility from memory
                recent_memories = market_state['memory'][-10:]
                if recent_memories:
                    vols = [m.get('volatility', 0) for m in recent_memories]
                    global_vol = np.mean(vols)
        
        if global_vol > 0.03:
            interval = 30  # More frequent in high volatility
        elif global_vol < 0.01:
            interval = 120  # Less frequent in low volatility
        
        return interval
    
    async def adapt_strategies(self):
        """Adapt strategies based on performance"""
        
        # Review strategy performance
        for strategy, performance in self.global_memory['strategy_performance'].items():
            if performance['trades'] >= 20:  # Enough trades to evaluate
                win_rate = performance['win_rate']
                
                # Adjust strategy usage
                if win_rate < 0.4:
                    # Poor performance - reduce usage
                    self.logger.info(f"Reducing usage of strategy {strategy} (win rate: {win_rate:.2%})")
                elif win_rate > 0.6:
                    # Good performance - increase usage
                    self.logger.info(f"Increasing usage of strategy {strategy} (win rate: {win_rate:.2%})")
    
    def calculate_position_size(self, symbol: str, confidence: float, account_info: Dict) -> float:
        """Calculate appropriate position size"""
        
        balance = account_info.get('balance', 10000)
        
        # Base position size (1% of account)
        base_size = balance * 0.01
        
        # Adjust based on confidence
        position_size = base_size * confidence
        
        # Market-specific adjustments
        if 'USD' in symbol:  # Forex
            # Forex typically uses lots
            lot_size = 100000  # Standard lot
            position_size = round(position_size / lot_size, 2)  # Mini lots
        else:  # Crypto
            # Crypto uses direct amounts
            position_size = round(position_size, 8)  # 8 decimal places
        
        return position_size


class ForexSpecialistBrain:
    """Specialized brain for Forex trading"""
    
    def __init__(self, main_brain):
        self.main_brain = main_brain
        self.forex_patterns = {
            'london_breakout': {'active_hours': (8, 10), 'success_rate': 0},
            'ny_session_trend': {'active_hours': (13, 17), 'success_rate': 0},
            'asian_range': {'active_hours': (0, 8), 'success_rate': 0}
        }
    
    async def analyze(self, observations: Dict) -> Dict:
        """Forex-specific analysis"""
        
        analysis = {
            'symbols': {},
            'session': self.get_trading_session(),
            'majors_correlation': {},
            'economic_impact': {}
        }
        
        # Analyze each forex pair
        for symbol, data in observations.get('market_data', {}).items():
            symbol_analysis = await self.analyze_forex_pair(symbol, data)
            analysis['symbols'][symbol] = symbol_analysis
        
        # Check major pairs correlation
        analysis['majors_correlation'] = self.analyze_major_pairs_correlation(
            analysis['symbols']
        )
        
        return analysis
    
    async def analyze_forex_pair(self, symbol: str, data: Dict) -> Dict:
        """Analyze individual forex pair"""
        
        analysis = {
            'trend_strength': 0,
            'trend_direction': 'neutral',
            'session_aligned': False,
            'indicators': {},
            'volatility': 0
        }
        
        # Get latest data
        df = data.get('H1')
        if df is None or len(df) < 100:
            return analysis
        
        # Calculate volatility
        analysis['volatility'] = df['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Forex-specific indicators
        # 1. Multi-timeframe trend
        h1_trend = self.calculate_trend(df)
        analysis['trend_strength'] = h1_trend['strength']
        analysis['trend_direction'] = h1_trend['direction']
        
        # 2. Session analysis
        current_session = self.get_trading_session()
        if self.is_pair_active_in_session(symbol, current_session):
            analysis['session_aligned'] = True
            analysis['trend_strength'] *= 1.2  # Boost for session alignment
        
        # 3. RSI
        analysis['indicators']['rsi'] = self.calculate_rsi(df['close'])
        
        # 4. ADX
        analysis['indicators']['adx'] = 25  # Simplified
        
        # 5. Check alignments
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
        
        if analysis['trend_direction'] == 'bullish' and df['close'].iloc[-1] > df['sma_20'].iloc[-1]:
            analysis['indicators']['price_ma_alignment'] = True
            analysis['indicators']['trend_bias'] = 1
        elif analysis['trend_direction'] == 'bearish' and df['close'].iloc[-1] < df['sma_20'].iloc[-1]:
            analysis['indicators']['price_ma_alignment'] = True
            analysis['indicators']['trend_bias'] = -1
        else:
            analysis['indicators']['price_ma_alignment'] = False
            analysis['indicators']['trend_bias'] = 0
        
        return analysis
    
    def calculate_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate trend for forex"""
        
        # Use multiple EMAs
        if 'close' not in df.columns:
            return {'direction': 'neutral', 'strength': 0}
        
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        # Determine trend direction
        if ema_20.iloc[-1] > ema_50.iloc[-1]:
            direction = 'bullish'
            strength = min(1.0, (ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1] * 100)
        elif ema_20.iloc[-1] < ema_50.iloc[-1]:
            direction = 'bearish'
            strength = min(1.0, (ema_50.iloc[-1] - ema_20.iloc[-1]) / ema_50.iloc[-1] * 100)
        else:
            direction = 'neutral'
            strength = 0.3
        
        return {'direction': direction, 'strength': abs(strength)}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def get_trading_session(self) -> str:
        """Get current forex trading session"""
        
        current_hour = datetime.now().hour
        
        if 0 <= current_hour < 8:
            return 'asian'
        elif 8 <= current_hour < 16:
            return 'london'
        elif 16 <= current_hour < 24:
            return 'newyork'
        
        return 'overlap'
    
    def is_pair_active_in_session(self, symbol: str, session: str) -> bool:
        """Check if pair is active in current session"""
        
        session_pairs = {
            'asian': ['USDJPY', 'AUDUSD', 'NZDUSD', 'AUDJPY'],
            'london': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY'],
            'newyork': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF']
        }
        
        return symbol in session_pairs.get(session, [])
    
    async def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """Find forex-specific opportunities"""
        
        opportunities = []
        
        # Session-based strategies
        current_session = self.get_trading_session()
        
        for symbol, analysis in market_data.get('symbols', {}).items():
            # London breakout strategy
            if current_session == 'london' and analysis.get('session_aligned'):
                if self.detect_breakout(analysis):
                    opportunities.append({
                        'type': 'session_breakout',
                        'symbol': symbol,
                        'direction': analysis['trend_direction'],
                        'strength': 0.8,
                        'strategy': 'london_breakout'
                    })
            
            # Trend continuation in active session
            if analysis.get('trend_strength', 0) > 0.6 and analysis.get('session_aligned'):
                opportunities.append({
                    'type': 'trend_continuation',
                    'symbol': symbol,
                    'direction': analysis['trend_direction'],
                    'strength': analysis['trend_strength'],
                    'strategy': 'session_trend'
                })
        
        return opportunities
    
    def detect_breakout(self, analysis: Dict) -> bool:
        """Detect breakout conditions"""
        
        # Check for volatility expansion
        if analysis.get('volatility', 0) > 0.015:
            # Check if trend is strong
            if analysis.get('trend_strength', 0) > 0.7:
                return True
        
        return False
    
    def analyze_major_pairs_correlation(self, symbols_analysis: Dict) -> Dict:
        """Analyze correlations between major forex pairs"""
        
        correlations = {}
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        for pair1 in major_pairs:
            for pair2 in major_pairs:
                if pair1 != pair2 and pair1 in symbols_analysis and pair2 in symbols_analysis:
                    # Simplified correlation based on trend direction
                    trend1 = symbols_analysis[pair1].get('trend_direction')
                    trend2 = symbols_analysis[pair2].get('trend_direction')
                    
                    if trend1 == trend2:
                        correlations[f"{pair1}_{pair2}"] = 0.5
                    else:
                        correlations[f"{pair1}_{pair2}"] = -0.3
        
        return correlations
    
    def get_trade_parameters(self, opportunity: Dict, score: float) -> Dict:
        """Get forex-specific trade parameters"""
        
        params = {
            'stop_loss_pct': 0.015,  # 1.5% stop loss
            'take_profit_pct': 0.03,  # 3% take profit
            'position_size_multiplier': 1.0
        }
        
        # Adjust based on opportunity type
        if opportunity['type'] == 'session_breakout':
            params['stop_loss_pct'] = 0.02
            params['take_profit_pct'] = 0.05
        elif opportunity['type'] == 'trend_continuation':
            params['stop_loss_pct'] = 0.01
            params['take_profit_pct'] = 0.025
        
        # Adjust based on score
        if score > 0.8:
            params['position_size_multiplier'] = 1.5
        elif score < 0.6:
            params['position_size_multiplier'] = 0.7
        
        return params


class CryptoSpecialistBrain:
    """Specialized brain for Cryptocurrency trading"""
    
    def __init__(self, main_brain):
        self.main_brain = main_brain
        self.crypto_patterns = {
            'btc_dominance': {'threshold': 0.45, 'impact': 'high'},
            'altcoin_season': {'indicators': [], 'active': False},
            'defi_momentum': {'tokens': ['UNI', 'AAVE', 'COMP'], 'strength': 0}
        }
    
    async def analyze(self, observations: Dict) -> Dict:
        """Crypto-specific analysis"""
        
        analysis = {
            'symbols': {},
            'btc_dominance': await self.analyze_btc_dominance(observations),
            'market_sentiment': self.analyze_crypto_sentiment(),
            'on_chain_metrics': {}
        }
        
        # Analyze each crypto pair
        for symbol, data in observations.get('market_data', {}).items():
            symbol_analysis = await self.analyze_crypto_pair(symbol, data)
            analysis['symbols'][symbol] = symbol_analysis
        
        return analysis
    
    async def analyze_crypto_pair(self, symbol: str, data: Dict) -> Dict:
        """Analyze individual crypto pair"""
        
        analysis = {
            'trend_strength': 0,
            'trend_direction': 'neutral',
            'volume_profile': 'normal',
            'indicators': {},
            'volatility': 0,
            'volatility_regime': 'normal'
        }
        
        # Get latest data
        df = data.get('1h')  # Use 1h for crypto
        if df is None or len(df) < 100:
            return analysis
        
        # Calculate volatility
        analysis['volatility'] = df['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Volatility regime
        if analysis['volatility'] > 0.04:
            analysis['volatility_regime'] = 'high'
        elif analysis['volatility'] < 0.02:
            analysis['volatility_regime'] = 'low'
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        if current_volume > avg_volume * 2:
            analysis['volume_profile'] = 'high'
        elif current_volume < avg_volume * 0.5:
            analysis['volume_profile'] = 'low'
        
        # Trend analysis
        trend = self.calculate_crypto_trend(df)
        analysis['trend_strength'] = trend['strength']
        analysis['trend_direction'] = trend['direction']
        
        # Crypto-specific indicators
        analysis['indicators']['rsi'] = self.calculate_rsi(df['close'])
        analysis['indicators']['volume_momentum'] = self.calculate_volume_momentum(df)
        
        # Check for pump conditions
        if self.detect_pump_conditions(df):
            analysis['pump_alert'] = True
            analysis['trend_strength'] *= 0.5  # Reduce confidence in pumps
        
        return analysis
    
    def calculate_crypto_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate trend for crypto with faster MAs"""
        
        if 'close' not in df.columns:
            return {'direction': 'neutral', 'strength': 0}
        
        # Use faster MAs for crypto
        ema_9 = df['close'].ewm(span=9).mean()
        ema_21 = df['close'].ewm(span=21).mean()
        
        # Determine trend direction
        if ema_9.iloc[-1] > ema_21.iloc[-1]:
            direction = 'bullish'
            strength = min(1.0, (ema_9.iloc[-1] - ema_21.iloc[-1]) / ema_21.iloc[-1] * 100)
        elif ema_9.iloc[-1] < ema_21.iloc[-1]:
            direction = 'bearish'
            strength = min(1.0, (ema_21.iloc[-1] - ema_9.iloc[-1]) / ema_21.iloc[-1] * 100)
        else:
            direction = 'neutral'
            strength = 0.3
        
        return {'direction': direction, 'strength': abs(strength)}
    
    def calculate_volume_momentum(self, df: pd.DataFrame) -> float:
        """Calculate volume momentum indicator"""
        
        if 'volume' not in df.columns:
            return 0
        
        # Volume moving average
        volume_ma = df['volume'].rolling(20).mean()
        
        # Recent volume vs average
        recent_vol = df['volume'].iloc[-5:].mean()
        avg_vol = volume_ma.iloc[-1]
        
        if avg_vol > 0:
            return (recent_vol - avg_vol) / avg_vol
        
        return 0
    
    def detect_pump_conditions(self, df: pd.DataFrame) -> bool:
        """Detect potential pump and dump conditions"""
        
        # Check for sudden price spike
        price_change_1h = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]
        
        # Check for volume spike
        volume_spike = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 3
        
        # Pump detected if large price move with volume spike
        return price_change_1h > 0.1 and volume_spike
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    async def analyze_btc_dominance(self, observations: Dict) -> Dict:
        """Analyze Bitcoin dominance and its impact"""
        
        btc_data = observations.get('market_data', {}).get('BTC/USDT', {})
        
        dominance_analysis = {
            'btc_trend': 'neutral',
            'alt_opportunity': 'neutral',
            'dominance_level': 0.45  # Default
        }
        
        if btc_data:
            btc_analysis = btc_data.get('1h')
            if btc_analysis is not None and len(btc_analysis) > 0:
                # Check BTC trend
                btc_trend = self.calculate_crypto_trend(btc_analysis)
                dominance_analysis['btc_trend'] = btc_trend['direction']
                
                # Determine alt opportunity
                if btc_trend['direction'] == 'neutral':
                    dominance_analysis['alt_opportunity'] = 'positive'
                elif btc_trend['direction'] == 'bearish' and btc_trend['strength'] > 0.7:
                    dominance_analysis['alt_opportunity'] = 'negative'
        
        return dominance_analysis
    
    def analyze_crypto_sentiment(self) -> str:
        """Analyze overall crypto market sentiment"""
        
        # Simplified sentiment analysis
        # In real implementation, this would use:
        # - Fear & Greed Index
        # - Social media sentiment
        # - Funding rates
        # - Open interest
        
        return 'neutral'  # Placeholder
    
    async def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """Find crypto-specific opportunities"""
        
        opportunities = []
        
        # BTC dominance impact
        btc_dominance = market_data.get('btc_dominance', {})
        
        for symbol, analysis in market_data.get('symbols', {}).items():
            # Volatility breakout opportunities
            if analysis.get('volatility_regime') == 'high' and \
               analysis.get('trend_strength', 0) > 0.6:
                opportunities.append({
                    'type': 'volatility_breakout',
                    'symbol': symbol,
                    'direction': analysis['trend_direction'],
                    'strength': min(0.9, analysis['trend_strength'] * 1.2),
                    'strategy': 'crypto_momentum'
                })
            
            # Low volatility accumulation
            if analysis.get('volatility_regime') == 'low':
                if 40 < analysis['indicators'].get('rsi', 50) < 60:
                    opportunities.append({
                        'type': 'accumulation',
                        'symbol': symbol,
                        'direction': 'buy',
                        'strength': 0.6,
                        'strategy': 'volatility_expansion'
                    })
            
            # DeFi momentum plays
            if 'DEFI' in symbol or symbol in ['UNI/USDT', 'AAVE/USDT', 'COMP/USDT']:
                if analysis['indicators'].get('rsi', 50) > 60:
                    opportunities.append({
                        'type': 'momentum',
                        'symbol': symbol,
                        'direction': 'buy',
                        'strength': 0.75,
                        'strategy': 'defi_momentum'
                    })
        
        return opportunities
    
    def get_trade_parameters(self, opportunity: Dict, score: float) -> Dict:
        """Get crypto-specific trade parameters"""
        
        params = {
            'stop_loss_pct': 0.03,  # 3% stop loss
            'take_profit_pct': 0.06,  # 6% take profit
            'position_size_multiplier': 1.0,
            'use_trailing_stop': True
        }
        
        # Adjust based on opportunity type
        if opportunity['type'] == 'volatility_breakout':
            params['stop_loss_pct'] = 0.05  # Wider stop for volatility
            params['take_profit_pct'] = 0.15  # Higher target
            params['use_trailing_stop'] = True
        elif opportunity['type'] == 'accumulation':
            params['stop_loss_pct'] = 0.02  # Tighter stop
            params['take_profit_pct'] = 0.04  # Lower target
            params['position_size_multiplier'] = 1.5  # Larger position for low vol
        
        # Adjust based on score
        if score > 0.8:
            params['position_size_multiplier'] *= 1.3
        
        return params


class ArbitrageBrain:
    """Specialized brain for cross-market arbitrage"""
    
    def __init__(self, main_brain):
        self.main_brain = main_brain
        self.arbitrage_pairs = {
            'forex_crypto': [
                ('EURUSD', 'EUR/USDT'),
                ('GBPUSD', 'GBP/USDT')
            ],
            'crypto_crypto': [
                ('BTC/USDT', 'BTC/USDC'),
                ('ETH/USDT', 'ETH/USDC')
            ]
        }
    
    async def find_opportunities(self, market_insights: Dict) -> List[Dict]:
        """Find arbitrage opportunities across markets"""
        
        opportunities = []
        
        # Check forex-crypto arbitrage
        if 'forex' in market_insights and 'crypto' in market_insights:
            forex_crypto_arb = self.find_forex_crypto_arbitrage(
                market_insights['forex'],
                market_insights['crypto']
            )
            opportunities.extend(forex_crypto_arb)
        
        # Check crypto-crypto arbitrage (between exchanges)
        if 'crypto' in market_insights:
            crypto_arb = self.find_crypto_arbitrage(market_insights['crypto'])
            opportunities.extend(crypto_arb)
        
        return opportunities
    
    def find_forex_crypto_arbitrage(self, forex_data: Dict, crypto_data: Dict) -> List[Dict]:
        """Find arbitrage between forex and crypto markets"""
        
        opportunities = []
        
        for forex_pair, crypto_pair in self.arbitrage_pairs['forex_crypto']:
            if forex_pair in forex_data.get('symbols', {}) and \
               crypto_pair in crypto_data.get('symbols', {}):
                
                # Calculate price discrepancy
                forex_price = self.get_normalized_price(forex_data['symbols'][forex_pair])
                crypto_price = self.get_normalized_price(crypto_data['symbols'][crypto_pair])
                
                discrepancy = abs(forex_price - crypto_price) / forex_price
                
                if discrepancy > 0.002:  # 0.2% discrepancy
                    opportunities.append({
                        'type': 'arbitrage',
                        'strategy': 'forex_crypto_arb',
                        'markets': ['forex', 'crypto'],
                        'symbols': [forex_pair, crypto_pair],
                        'direction': 'buy_forex_sell_crypto' if forex_price < crypto_price else 'buy_crypto_sell_forex',
                        'expected_profit': discrepancy,
                        'confidence': min(0.9, discrepancy * 100)
                    })
        
        return opportunities
    
    def find_crypto_arbitrage(self, crypto_data: Dict) -> List[Dict]:
        """Find arbitrage opportunities within crypto"""
        
        opportunities = []
        
        # Check for stable coin arbitrage
        for base_pair, quote_pair in self.arbitrage_pairs['crypto_crypto']:
            if base_pair in crypto_data.get('symbols', {}) and \
               quote_pair in crypto_data.get('symbols', {}):
                
                # Price comparison
                base_price = self.get_normalized_price(crypto_data['symbols'][base_pair])
                quote_price = self.get_normalized_price(crypto_data['symbols'][quote_pair])
                
                if base_price > 0 and quote_price > 0:
                    discrepancy = abs(base_price - quote_price) / base_price
                    
                    if discrepancy > 0.001:  # 0.1% for crypto
                        opportunities.append({
                            'type': 'arbitrage',
                            'strategy': 'crypto_arb',
                            'markets': ['crypto'],
                            'symbols': [base_pair, quote_pair],
                            'direction': 'buy_low_sell_high',
                            'expected_profit': discrepancy,
                            'confidence': min(0.85, discrepancy * 200)
                        })
        
        return opportunities
    
    def get_normalized_price(self, symbol_data: Dict) -> float:
        """Get normalized price from symbol data"""
        
        # Try to get the current price
        if 'current_price' in symbol_data:
            return symbol_data['current_price']
        
        # Otherwise use trend data
        if 'trend_strength' in symbol_data:
            # This is simplified - in real implementation would use actual price
            return 1.0 + (symbol_data.get('trend_strength', 0) * 0.01)
        
        return 0