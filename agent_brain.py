import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import asyncio
import logging

class AgentBrain:
    """AI Agent's cognitive system"""
    
    def __init__(self, system):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
        # Agent state
        self.risk_level = 'moderate'
        self.active_strategies = []
        self.learning_progress = 0.0
        
        # Memory
        self.short_term_memory = []  # Recent observations
        self.long_term_memory = {}   # Learned patterns
        self.experience_buffer = []  # Trade results
        
        # Decision parameters
        self.thinking_speed = 1.0    # Multiplier for thinking interval
        self.risk_tolerance = 0.5    # 0-1 scale
        self.exploration_rate = 0.1  # For trying new strategies
    
    async def think(self, observations: Dict) -> List[Dict]:
        """Main thinking process"""
        
        # Store observation
        self.short_term_memory.append(observations)
        if len(self.short_term_memory) > 100:
            self.short_term_memory.pop(0)
        
        # Analyze situation
        situation = self.analyze_situation(observations)
        
        # Generate possible actions
        possible_actions = self.generate_actions(situation)
        
        # Evaluate and select actions
        selected_actions = self.evaluate_actions(possible_actions, situation)
        
        # Create decisions
        decisions = []
        for action in selected_actions:
            decision = self.create_decision(action, situation)
            decisions.append(decision)
        
        return decisions
    
    def analyze_situation(self, observations: Dict) -> Dict:
        """Analyze current market situation"""
        
        situation = {
            'market_state': 'normal',
            'volatility': 'medium',
            'trend': 'neutral',
            'opportunities': [],
            'threats': []
        }
        
        # Analyze each symbol's data
        for symbol, data in observations.get('market_data', {}).items():
            if 'H1' in data or '1h' in data:
                df = data.get('H1', data.get('1h'))
                
                # Calculate indicators
                returns = df['close'].pct_change()
                volatility = returns.std()
                trend = np.polyfit(range(len(df)), df['close'], 1)[0]
                
                # Classify market state
                if volatility > 0.02:
                    situation['volatility'] = 'high'
                elif volatility < 0.005:
                    situation['volatility'] = 'low'
                
                if trend > 0.0001:
                    situation['trend'] = 'bullish'
                elif trend < -0.0001:
                    situation['trend'] = 'bearish'
                
                # Look for opportunities
                if self.detect_opportunity(df, symbol):
                    situation['opportunities'].append({
                        'symbol': symbol,
                        'type': 'breakout',
                        'confidence': 0.7
                    })
        
        return situation
    
    def generate_actions(self, situation: Dict) -> List[Dict]:
        """Generate possible actions based on situation"""
        
        actions = []
        
        # Trading actions
        for opportunity in situation['opportunities']:
            actions.append({
                'type': 'trade',
                'symbol': opportunity['symbol'],
                'direction': 'buy' if situation['trend'] == 'bullish' else 'sell',
                'confidence': opportunity['confidence']
            })
        
        # Risk management actions
        if situation['volatility'] == 'high':
            actions.append({
                'type': 'adjust_risk',
                'parameters': {'risk_level': 'conservative'}
            })
        
        # Portfolio actions
        positions = self.system.positions
        if len(positions) > 5:
            actions.append({
                'type': 'rebalance',
                'reason': 'too_many_positions'
            })
        
        return actions
    
    def evaluate_actions(self, actions: List[Dict], situation: Dict) -> List[Dict]:
        """Evaluate and filter actions"""
        
        evaluated_actions = []
        
        for action in actions:
            score = self.score_action(action, situation)
            
            if score > 0.6:  # Threshold
                action['score'] = score
                evaluated_actions.append(action)
        
        # Sort by score and limit number of actions
        evaluated_actions.sort(key=lambda x: x['score'], reverse=True)
        
        return evaluated_actions[:3]  # Max 3 actions at a time
    
    def score_action(self, action: Dict, situation: Dict) -> float:
        """Score an action based on various factors"""
        
        score = 0.5  # Base score
        
        # Adjust based on action type
        if action['type'] == 'trade':
                        # Check if aligns with trend
            if (action['direction'] == 'buy' and situation['trend'] == 'bullish') or \
               (action['direction'] == 'sell' and situation['trend'] == 'bearish'):
                score += 0.2
            
            # Adjust for confidence
            score += action.get('confidence', 0) * 0.3
            
            # Penalize if high volatility and conservative
            if situation['volatility'] == 'high' and self.risk_level == 'conservative':
                score -= 0.3
        
        elif action['type'] == 'adjust_risk':
            # Favor risk adjustment in volatile markets
            if situation['volatility'] == 'high':
                score += 0.3
        
        elif action['type'] == 'rebalance':
            # Favor rebalancing when needed
            score += 0.4
        
        # Apply exploration bonus
        if np.random.random() < self.exploration_rate:
            score += 0.1
        
        return max(0, min(1, score))  # Clamp between 0 and 1
    
    def create_decision(self, action: Dict, situation: Dict) -> Dict:
        """Create a decision from an action"""
        
        decision = {
            'timestamp': datetime.now(),
            'type': action['type'],
            'confidence': action.get('score', 0.5),
            'situation': situation
        }
        
        # Add specific parameters based on action type
        if action['type'] == 'trade':
            decision.update({
                'symbol': action['symbol'],
                'direction': action['direction'],
                'stop_loss_pct': 0.02 if situation['volatility'] == 'high' else 0.015,
                'take_profit_pct': 0.04 if situation['volatility'] == 'high' else 0.03
            })
        
        elif action['type'] == 'adjust_risk':
            decision.update({
                'parameters': action['parameters']
            })
        
        return decision
    
    def detect_opportunity(self, df: pd.DataFrame, symbol: str) -> bool:
        """Detect trading opportunity in data"""
        
        if len(df) < 50:
            return False
        
        # Simple breakout detection
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        
        current_price = df['close'].iloc[-1]
        
        # Bullish breakout
        if current_price > high_20.iloc[-2]:
            return True
        
        # Bearish breakout
        if current_price < low_20.iloc[-2]:
            return True
        
        return False
    
    def calculate_position_size(self, symbol: str, confidence: float, account_info: Dict) -> float:
        """Calculate position size using Kelly Criterion"""
        
        # Get win rate from experience
        win_rate = self.get_win_rate(symbol)
        
        # Average win/loss ratio
        avg_win = 0.03  # 3% average win
        avg_loss = 0.015  # 1.5% average loss
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b
        
        # Apply safety factor and confidence
        position_fraction = kelly_fraction * 0.25 * confidence
        
        # Apply risk level multiplier
        if self.risk_level == 'conservative':
            position_fraction *= 0.5
        elif self.risk_level == 'aggressive':
            position_fraction *= 1.5
        
        # Calculate actual position size
        account_balance = account_info.get('balance', 10000)
        position_value = account_balance * max(0.001, min(0.05, position_fraction))
        
        # Convert to lots (simplified)
        if self.system.active_market == 'forex':
            return round(position_value / 10000, 2)  # Standard lot = 100,000
        else:
            # For crypto, return portion of balance
            return position_value / 1000  # Adjust based on symbol
    
    def get_win_rate(self, symbol: str) -> float:
        """Get historical win rate for symbol"""
        
        symbol_trades = [t for t in self.experience_buffer if t.get('symbol') == symbol]
        
        if len(symbol_trades) < 10:
            return 0.5  # Default 50%
        
        wins = sum(1 for t in symbol_trades if t.get('profit', 0) > 0)
        
        return wins / len(symbol_trades)
    
    def record_trade(self, decision: Dict, result: Dict):
        """Record trade result for learning"""
        
        trade_record = {
            'timestamp': datetime.now(),
            'decision': decision,
            'result': result,
            'symbol': decision.get('symbol'),
            'profit': 0  # Will be updated when trade closes
        }
        
        self.experience_buffer.append(trade_record)
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > 1000:
            self.experience_buffer.pop(0)
    
    async def learn_from_experience(self):
        """Learn from trading experience"""
        
        if len(self.experience_buffer) < 50:
            return
        
        # Analyze recent trades
        recent_trades = self.experience_buffer[-100:]
        
        # Calculate performance metrics
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Update risk level based on performance
        if win_rate < 0.4:
            self.risk_level = 'conservative'
            self.logger.info("Switching to conservative risk due to low win rate")
        elif win_rate > 0.6:
            self.risk_level = 'moderate'
            self.logger.info("Switching to moderate risk due to good win rate")
        
        # Update learning progress
        self.learning_progress = min(1.0, len(self.experience_buffer) / 500)
        
        # Extract patterns from successful trades
        self.extract_patterns()
    
    def extract_patterns(self):
        """Extract patterns from successful trades"""
        
        successful_trades = [t for t in self.experience_buffer if t.get('profit', 0) > 0.01]
        
        if len(successful_trades) < 20:
            return
        
        # Group by situation characteristics
        patterns = {}
        
        for trade in successful_trades:
            situation = trade.get('decision', {}).get('situation', {})
            
            pattern_key = f"{situation.get('trend')}_{situation.get('volatility')}"
            
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            
            patterns[pattern_key].append(trade)
        
        # Store patterns with success rates
        for pattern_key, trades in patterns.items():
            success_rate = len(trades) / len([t for t in self.experience_buffer 
                                            if f"{t.get('decision', {}).get('situation', {}).get('trend')}_{t.get('decision', {}).get('situation', {}).get('volatility')}" == pattern_key])
            
            self.long_term_memory[pattern_key] = {
                'success_rate': success_rate,
                'sample_size': len(trades),
                'avg_profit': np.mean([t.get('profit', 0) for t in trades])
            }
    
    async def adapt_strategies(self):
        """Adapt trading strategies based on performance"""
        
        # Evaluate current strategies
        strategy_performance = self.evaluate_strategies()
        
        # Remove underperforming strategies
        for strategy in self.active_strategies[:]:
            if strategy_performance.get(strategy, {}).get('score', 0) < 0.3:
                self.active_strategies.remove(strategy)
                self.logger.info(f"Removed underperforming strategy: {strategy}")
        
        # Add new strategies if needed
        if len(self.active_strategies) < 3:
            new_strategy = self.generate_new_strategy()
            if new_strategy:
                self.active_strategies.append(new_strategy)
                self.logger.info(f"Added new strategy: {new_strategy}")
    
    def evaluate_strategies(self) -> Dict:
        """Evaluate performance of active strategies"""
        
        performance = {}
        
        # Simplified evaluation - in reality would be more complex
        for strategy in self.active_strategies:
            # Mock evaluation
            performance[strategy] = {
                'score': np.random.random(),
                'trades': 10,
                'win_rate': 0.5
            }
        
        return performance
    
    def generate_new_strategy(self) -> str:
        """Generate a new trading strategy"""
        
        # Based on successful patterns
        if self.long_term_memory:
            best_pattern = max(self.long_term_memory.items(), 
                             key=lambda x: x[1]['success_rate'])
            
            # Create strategy based on pattern
            if 'bullish_high' in best_pattern[0]:
                return 'momentum_breakout'
            elif 'bearish_low' in best_pattern[0]:
                return 'mean_reversion'
        
        # Default strategies
        available_strategies = [
            'trend_following',
            'breakout',
            'mean_reversion',
            'momentum'
        ]
        
        for strategy in available_strategies:
            if strategy not in self.active_strategies:
                return strategy
        
        return None
    
    def get_thinking_interval(self) -> float:
        """Determine how often to think based on market conditions"""
        
        base_interval = 60  # 1 minute
        
        # Adjust based on volatility
        if len(self.short_term_memory) > 0:
            last_observation = self.short_term_memory[-1]
            
            # Check volatility across symbols
            high_volatility_count = 0
            
            for symbol_data in last_observation.get('market_data', {}).values():
                if 'H1' in symbol_data or '1h' in symbol_data:
                    df = symbol_data.get('H1', symbol_data.get('1h'))
                    volatility = df['close'].pct_change().std()
                    
                    if volatility > 0.02:
                        high_volatility_count += 1
            
            # Think more frequently in volatile markets
            if high_volatility_count > 2:
                base_interval *= 0.5
        
        # Apply thinking speed multiplier
        return base_interval * self.thinking_speed
    
    def adjust_risk_parameters(self, level: str):
        """Adjust risk parameters"""
        
        self.risk_level = level
        
        if level == 'conservative':
            self.risk_tolerance = 0.3
            self.exploration_rate = 0.05
        elif level == 'moderate':
            self.risk_tolerance = 0.5
            self.exploration_rate = 0.1
        elif level == 'aggressive':
            self.risk_tolerance = 0.7
            self.exploration_rate = 0.15
        
        self.logger.info(f"Adjusted risk parameters to {level}")