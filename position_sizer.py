import numpy as np
from typing import Dict, Optional
import logging

class PositionSizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, signal_data: Dict, account_data: Dict,
                              market_data: Dict) -> float:
        """Calculate optimal position size"""
        # Get base position size methods
        fixed_fractional = self.fixed_fractional_size(account_data)
        kelly_size = self.kelly_criterion_size(signal_data, account_data)
        volatility_based = self.volatility_based_size(account_data, market_data)
        optimal_f = self.optimal_f_size(signal_data, account_data)
        
        # Combine methods with weights
        weights = self.config.get('position_size_weights', {
            'fixed_fractional': 0.25,
            'kelly': 0.25,
            'volatility': 0.25,
            'optimal_f': 0.25
        })
        
        combined_size = (
            fixed_fractional * weights['fixed_fractional'] +
            kelly_size * weights['kelly'] +
            volatility_based * weights['volatility'] +
            optimal_f * weights['optimal_f']
        )
        
        # Apply constraints
        final_size = self.apply_constraints(combined_size, account_data, market_data)
        
        return final_size
    
    def fixed_fractional_size(self, account_data: Dict) -> float:
        """Fixed fractional position sizing"""
        risk_per_trade = self.config.get('risk_per_trade', 0.02)
        return account_data['balance'] * risk_per_trade
    
    def kelly_criterion_size(self, signal_data: Dict, account_data: Dict) -> float:
        """Kelly criterion position sizing"""
        win_prob = signal_data.get('win_probability', 0.55)
        win_loss_ratio = signal_data.get('win_loss_ratio', 1.5)
        
        # Kelly formula
        kelly_percent = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Apply Kelly fraction
        kelly_fraction = self.config.get('kelly_fraction', 0.25)
        position_percent = kelly_percent * kelly_fraction
        
        # Ensure positive and reasonable
        position_percent = max(0, min(position_percent, 0.2))
        
        return account_data['balance'] * position_percent
    
    def volatility_based_size(self, account_data: Dict, market_data: Dict) -> float:
        """Volatility-based position sizing"""
        target_risk = self.config.get('target_risk', 0.02)
        current_volatility = market_data.get('volatility', 0.02)
        
        # Adjust position size inversely to volatility
        volatility_scalar = target_risk / (current_volatility + 1e-10)
        base_size = account_data['balance'] * target_risk
        
        return base_size * min(volatility_scalar, 2.0)  # Cap at 2x
    
    def optimal_f_size(self, signal_data: Dict, account_data: Dict) -> float:
        """Optimal f position sizing"""
        # Get historical trade results
        trade_results = signal_data.get('historical_trades', [])
        
        if len(trade_results) < 20:
            # Not enough data, use fixed fractional
            return self.fixed_fractional_size(account_data)
        
        # Calculate optimal f
        optimal_f = self.calculate_optimal_f(trade_results)
        
        return account_data['balance'] * optimal_f
    
    def calculate_optimal_f(self, trade_results: list) -> float:
        """Calculate optimal f from trade history"""
        if not trade_results:
            return 0.02
        
        # Convert to returns
        returns = np.array(trade_results)
        
        # Search for optimal f
        f_values = np.arange(0.01, 0.5, 0.01)
        twrs = []
        
        for f in f_values:
            twr = 1.0
            for ret in returns:
                twr *= (1 + f * ret)
            twrs.append(twr)
        
        # Find f that maximizes TWR
        optimal_idx = np.argmax(twrs)
        optimal_f = f_values[optimal_idx]
        
        # Apply safety factor
        return optimal_f * 0.5
    
    def apply_constraints(self, position_size: float, account_data: Dict,
                         market_data: Dict) -> float:
        """Apply position size constraints"""
        # Maximum position size
        max_position = account_data['balance'] * self.config.get('max_position_size', 0.1)
        position_size = min(position_size, max_position)
        
        # Minimum position size
        min_position = self.config.get('min_position_size', 0.01)
        if position_size < min_position:
            return 0  # Don't trade if below minimum
        
        # Adjust for leverage
        max_leverage = self.config.get('max_leverage', 10)
        if position_size > account_data['free_margin'] * max_leverage:
            position_size = account_data['free_margin'] * max_leverage
        
        # Adjust for correlation
        correlation_adjustment = self.correlation_adjustment(market_data)
        position_size *= correlation_adjustment
        
        # Round to valid lot size
        position_size = self.round_to_lot_size(position_size, market_data)
        
        return position_size
    
    def correlation_adjustment(self, market_data: Dict) -> float:
        """Adjust size based on portfolio correlation"""
        avg_correlation = market_data.get('portfolio_correlation', 0)
        
        if avg_correlation > 0.8:
            return 0.5
        elif avg_correlation > 0.6:
            return 0.7
        elif avg_correlation > 0.4:
            return 0.85
        else:
            return 1.0
    
    def round_to_lot_size(self, position_size: float, market_data: Dict) -> float:
        """Round position size to valid lot size"""
        lot_step = market_data.get('lot_step', 0.01)
        min_lot = market_data.get('min_lot', 0.01)
        
        # Round to nearest lot step
        lots = max(min_lot, round(position_size / lot_step) * lot_step)
        
        return lots