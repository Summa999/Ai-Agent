import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class MarketInterface(ABC):
    """Abstract base class for market interfaces"""
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str):
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, amount: float, price: float = None):
        pass
    
    @abstractmethod
    async def get_balance(self):
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int):
        pass

class AITradingAgent:
    """Autonomous AI Trading Agent with multi-market support"""
    
    def __init__(self, config_path='agent_config.yaml'):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger('AIAgent')
        
        # Agent State
        self.state = {
            'mode': 'learning',  # learning, trading, optimizing
            'risk_level': 'moderate',  # conservative, moderate, aggressive
            'active_strategies': [],
            'performance_history': [],
            'learning_progress': 0,
            'total_trades': 0,
            'profitable_trades': 0
        }
        
        # Agent Memory
        self.memory = {
            'successful_patterns': [],
            'failed_patterns': [],
            'market_conditions': {},
            'strategy_performance': {}
        }
        
        # Multi-market interfaces
        self.markets = {}
        self.initialize_markets()
        
        # AI Components
        self.decision_engine = DecisionEngine(self)
        self.strategy_manager = StrategyManager(self)
        self.risk_manager = AdaptiveRiskManager(self)
        self.learning_module = ContinuousLearning(self)
        
        # Agent personality/behavior
        self.personality = self.config.get('agent_personality', {
            'risk_tolerance': 0.5,
            'patience': 0.7,
            'aggression': 0.3,
            'adaptability': 0.8
        })
        
    def initialize_markets(self):
        """Initialize connections to multiple markets"""
        
        # Forex (MT5)
        if self.config.get('markets', {}).get('forex', {}).get('enabled'):
            from connectors.mt5_connector import MT5Interface
            self.markets['forex'] = MT5Interface(self.config['markets']['forex'])
        
        # Crypto
        if self.config.get('markets', {}).get('crypto', {}).get('enabled'):
            from connectors.crypto_connector import CryptoInterface
            self.markets['crypto'] = CryptoInterface(self.config['markets']['crypto'])
        
        # Stocks
        if self.config.get('markets', {}).get('stocks', {}).get('enabled'):
            from connectors.stock_connector import StockInterface
            self.markets['stocks'] = StockInterface(self.config['markets']['stocks'])
    
    async def think(self):
        """Main thinking/decision loop of the agent"""
        
        while True:
            try:
                # 1. Observe - Gather information from all markets
                observations = await self.observe_markets()
                
                # 2. Orient - Analyze situation and update beliefs
                situation_analysis = await self.analyze_situation(observations)
                
                # 3. Decide - Make decisions based on analysis
                decisions = await self.decision_engine.make_decisions(situation_analysis)
                
                # 4. Act - Execute decisions
                for decision in decisions:
                    await self.execute_decision(decision)
                
                # 5. Learn - Update knowledge from results
                await self.learning_module.update_from_experience()
                
                # 6. Adapt - Modify strategies based on performance
                await self.adapt_strategies()
                
                # Sleep based on agent state
                sleep_time = self.get_thinking_interval()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in agent thinking: {e}")
                await self.handle_error(e)
    
    async def observe_markets(self) -> Dict[str, Any]:
        """Gather information from all active markets"""
        observations = {
            'timestamp': datetime.now(),
            'markets': {},
            'global_indicators': {}
        }
        
        # Gather data from each market
        for market_name, market_interface in self.markets.items():
            if market_interface.is_connected():
                market_data = await market_interface.get_market_state()
                observations['markets'][market_name] = market_data
        
        # Add global indicators (VIX, DXY, etc.)
        observations['global_indicators'] = await self.get_global_indicators()
        
        return observations
    
    async def analyze_situation(self, observations: Dict) -> Dict:
        """Comprehensive situation analysis"""
        analysis = {
            'market_regime': {},
            'opportunities': [],
            'risks': [],
            'correlations': {},
            'sentiment': {}
        }
        
        # Analyze each market
        for market_name, market_data in observations['markets'].items():
            # Determine market regime
            regime = await self.detect_market_regime(market_data)
            analysis['market_regime'][market_name] = regime
            
            # Find opportunities
            opportunities = await self.find_opportunities(market_name, market_data)
            analysis['opportunities'].extend(opportunities)
            
            # Assess risks
            risks = await self.assess_risks(market_name, market_data)
            analysis['risks'].extend(risks)
        
        # Cross-market analysis
        analysis['correlations'] = await self.analyze_correlations(observations)
        analysis['sentiment'] = await self.analyze_sentiment()
        
        return analysis
    
    async def execute_decision(self, decision: Dict):
        """Execute a trading decision"""
        
        decision_type = decision['type']
        
        if decision_type == 'open_position':
            await self.open_position(decision)
            
        elif decision_type == 'close_position':
            await self.close_position(decision)
            
        elif decision_type == 'adjust_risk':
            await self.risk_manager.adjust_parameters(decision['params'])
            
        elif decision_type == 'change_strategy':
            await self.strategy_manager.switch_strategy(decision['strategy'])
            
        elif decision_type == 'hedge':
            await self.create_hedge(decision)
            
        elif decision_type == 'rebalance':
            await self.rebalance_portfolio(decision)
    
    async def adapt_strategies(self):
        """Adapt strategies based on performance"""
        
        # Evaluate current strategies
        performance_metrics = await self.evaluate_strategies()
        
        # Identify underperforming strategies
        for strategy_id, metrics in performance_metrics.items():
            if metrics['sharpe_ratio'] < self.config['min_sharpe_ratio']:
                # Either optimize or replace strategy
                if metrics['sample_size'] > 50:
                    await self.strategy_manager.replace_strategy(strategy_id)
                else:
                    await self.strategy_manager.optimize_strategy(strategy_id)
        
        # Explore new strategies if needed
        if self.should_explore_new_strategies():
            await self.strategy_manager.generate_new_strategy()
    
    def get_thinking_interval(self) -> float:
        """Determine how often the agent should think"""
        
        base_interval = 60  # 1 minute base
        
        # Adjust based on market volatility
        volatility_multiplier = self.get_volatility_multiplier()
        
        # Adjust based on agent state
        if self.state['mode'] == 'learning':
            state_multiplier = 0.5  # Think more frequently when learning
        elif self.state['mode'] == 'trading':
            state_multiplier = 1.0
        else:
            state_multiplier = 2.0  # Think less frequently when optimizing
        
        return base_interval * volatility_multiplier * state_multiplier
    
    async def report_status(self):
        """Generate and send status report"""
        
        report = f"""
        🤖 AI Agent Status Report
        
        Mode: {self.state['mode']}
        Risk Level: {self.state['risk_level']}
        Active Markets: {list(self.markets.keys())}
        Active Strategies: {len(self.state['active_strategies'])}
        
        Performance:
        - Total Trades: {self.state['total_trades']}
        - Win Rate: {self.state['profitable_trades'] / max(1, self.state['total_trades']):.2%}
        - Learning Progress: {self.state['learning_progress']:.1%}
        
        Current Positions: {await self.get_total_positions()}
        Total Equity: ${await self.get_total_equity():.2f}
        """
        
        return report

class DecisionEngine:
    """Makes autonomous trading decisions"""
    
    def __init__(self, agent):
        self.agent = agent
        self.decision_models = {}
        self.load_decision_models()
    
    async def make_decisions(self, analysis: Dict) -> List[Dict]:
        """Make trading decisions based on analysis"""
        
        decisions = []
        
        # 1. Risk Management Decisions
        risk_decisions = await self.make_risk_decisions(analysis)
        decisions.extend(risk_decisions)
        
        # 2. Trading Decisions
        trading_decisions = await self.make_trading_decisions(analysis)
        decisions.extend(trading_decisions)
        
        # 3. Portfolio Decisions
        portfolio_decisions = await self.make_portfolio_decisions(analysis)
        decisions.extend(portfolio_decisions)
        
        # 4. Strategy Decisions
        strategy_decisions = await self.make_strategy_decisions(analysis)
        decisions.extend(strategy_decisions)
        
        # Filter and prioritize decisions
        decisions = self.filter_decisions(decisions)
        decisions = self.prioritize_decisions(decisions)
        
        return decisions
    
    async def make_trading_decisions(self, analysis: Dict) -> List[Dict]:
        """Generate trading decisions"""
        
        decisions = []
        
        for opportunity in analysis['opportunities']:
            # Evaluate opportunity
            score = await self.evaluate_opportunity(opportunity)
            
            if score > self.agent.config['min_opportunity_score']:
                # Check if we should take this trade
                if await self.should_take_trade(opportunity, analysis):
                    decision = {
                        'type': 'open_position',
                        'market': opportunity['market'],
                        'symbol': opportunity['symbol'],
                        'side': opportunity['side'],
                        'confidence': score,
                        'strategy': opportunity['strategy'],
                        'risk_reward': opportunity['risk_reward'],
                        'timestamp': datetime.now()
                    }
                    decisions.append(decision)
        
        return decisions

class StrategyManager:
    """Manages and evolves trading strategies"""
    
    def __init__(self, agent):
        self.agent = agent
        self.strategies = {}
        self.strategy_generator = StrategyGenerator()
    
    async def generate_new_strategy(self):
        """Generate a new trading strategy using AI"""
        
        # Analyze what's working in current market
        market_analysis = await self.analyze_market_conditions()
        
        # Generate strategy parameters
        strategy_params = await self.strategy_generator.generate(
            market_analysis,
            self.agent.memory['successful_patterns']
        )
        
        # Backtest the strategy
        backtest_results = await self.backtest_strategy(strategy_params)
        
        if backtest_results['sharpe_ratio'] > 1.0:
            # Add to active strategies
            strategy_id = f"strategy_{len(self.strategies)}"
            self.strategies[strategy_id] = strategy_params
            self.agent.state['active_strategies'].append(strategy_id)

class AdaptiveRiskManager:
    """Dynamically adjusts risk based on market conditions and performance"""
    
    def __init__(self, agent):
        self.agent = agent
        self.risk_models = {}
        self.current_risk_params = {
            'max_position_size': 0.02,
            'max_daily_loss': 0.05,
            'max_correlation': 0.7,
            'var_limit': 0.03
        }
    
    async def adjust_parameters(self, params: Dict):
        """Adjust risk parameters dynamically"""
        
        # Validate new parameters
        validated_params = self.validate_params(params)
        
        # Gradually adjust (avoid sudden changes)
        for param, value in validated_params.items():
            current = self.current_risk_params.get(param, 0)
            # Smooth adjustment
            new_value = current * 0.7 + value * 0.3
            self.current_risk_params[param] = new_value

class ContinuousLearning:
    """Handles continuous learning and improvement"""
    
    def __init__(self, agent):
        self.agent = agent
        self.experience_buffer = []
        self.learning_rate = 0.001
    
    async def update_from_experience(self):
        """Learn from recent trading experience"""
        
        if len(self.experience_buffer) > 100:
            # Train models with new data
            await self.retrain_models()
            
            # Update successful patterns
            await self.extract_patterns()
            
            # Clear old experiences
            self.experience_buffer = self.experience_buffer[-1000:]