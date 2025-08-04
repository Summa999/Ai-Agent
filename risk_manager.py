import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    def __init__(self, config):
        self.config = config
        self.risk_limits = config.get('risk_limits', {})
        self.portfolio_history = []
        self.var_history = []
        
    def calculate_position_size(self, signal_strength, account_info, market_conditions):
        """Calculate optimal position size using Kelly Criterion and risk parity"""
        base_size = self.kelly_criterion(signal_strength, account_info)
        
        # Adjust for market conditions
        regime_multiplier = market_conditions.get('regime_rules', {}).get('position_size_multiplier', 1.0)
        
        # Volatility adjustment
        vol_adjustment = self.volatility_adjustment(market_conditions)
        
        # Correlation adjustment
        correlation_adjustment = self.correlation_adjustment(market_conditions)
        
        # Final position size
        position_size = base_size * regime_multiplier * vol_adjustment * correlation_adjustment
        
        # Apply limits
        max_position = account_info['balance'] * self.risk_limits.get('max_position_size', 0.1)
        position_size = min(position_size, max_position)
        
        return position_size
    
    def kelly_criterion(self, signal_strength, account_info):
        """Calculate position size using Kelly Criterion"""
        # Estimate win probability and win/loss ratio from historical data
        win_prob = signal_strength.get('win_probability', 0.55)
        win_loss_ratio = signal_strength.get('profit_factor', 1.5)
        
        # Kelly formula: f = (p * b - q) / b
        # where f = fraction of capital to bet
        # p = probability of win
        # q = probability of loss (1 - p)
        # b = win/loss ratio
        
        q = 1 - win_prob
        kelly_fraction = (win_prob * win_loss_ratio - q) / win_loss_ratio
        
        # Apply Kelly fraction with safety factor
        safety_factor = self.config.get('kelly_safety_factor', 0.25)
        position_fraction = kelly_fraction * safety_factor
        
        # Ensure positive and reasonable size
        position_fraction = max(0, min(position_fraction, 0.2))
        
        return account_info['balance'] * position_fraction
    
    def volatility_adjustment(self, market_conditions):
        """Adjust position size based on volatility"""
        current_vol = market_conditions.get('volatility', {}).get('current', 0.02)
        target_vol = self.config.get('target_volatility', 0.02)
        
        # Inverse volatility scaling
        vol_adjustment = min(target_vol / (current_vol + 1e-10), 2.0)
        
        return vol_adjustment
    
    def correlation_adjustment(self, market_conditions):
        """Adjust for portfolio correlation"""
        avg_correlation = market_conditions.get('portfolio_correlation', 0.3)
        
        # Reduce size when correlations are high
        if avg_correlation > 0.7:
            return 0.5
        elif avg_correlation > 0.5:
            return 0.75
        else:
            return 1.0
    
    def calculate_var(self, portfolio_returns, confidence_level=0.95):
        """Calculate Value at Risk using multiple methods"""
        # Historical VaR
        historical_var = self.historical_var(portfolio_returns, confidence_level)
        
        # Parametric VaR
        parametric_var = self.parametric_var(portfolio_returns, confidence_level)
        
        # Monte Carlo VaR
        monte_carlo_var = self.monte_carlo_var(portfolio_returns, confidence_level)
        
        # Conditional VaR (CVaR)
        cvar = self.calculate_cvar(portfolio_returns, confidence_level)
        
        var_results = {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'monte_carlo_var': monte_carlo_var,
            'cvar': cvar,
            'combined_var': np.mean([historical_var, parametric_var, monte_carlo_var])
        }
        
        self.var_history.append({
            'timestamp': pd.Timestamp.now(),
            'var_results': var_results
        })
        
        return var_results
    
    def historical_var(self, returns, confidence_level):
        """Calculate historical VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def parametric_var(self, returns, confidence_level):
        """Calculate parametric VaR assuming normal distribution"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Use normal distribution
        var = mean + std * norm.ppf(1 - confidence_level)
        
        return var
    
    def monte_carlo_var(self, returns, confidence_level, n_simulations=10000):
        """Calculate VaR using Monte Carlo simulation"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate simulations
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return var
    
    def calculate_cvar(self, returns, confidence_level):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.historical_var(returns, confidence_level)
        
        # CVaR is the average of returns below VaR
        cvar = np.mean(returns[returns <= var])
        
        return cvar
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if len(excess_returns) < 2:
            return 0
        
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-10) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sortino ratio (uses downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        
        # Downside returns only
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return 0
        
        downside_std = np.std(downside_returns)
        
        return np.mean(excess_returns) / (downside_std + 1e-10) * np.sqrt(252)
    
    def calculate_calmar_ratio(self, returns, period_years=3):
        """Calculate Calmar ratio (return / max drawdown)"""
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return annualized_return / (abs(max_drawdown) + 1e-10)
    
    def dynamic_stop_loss(self, entry_price, current_price, volatility, position_type='long'):
        """Calculate dynamic stop loss based on volatility"""
        # Base stop loss
        base_stop_percent = self.config.get('base_stop_loss', 0.02)
        
        # Adjust for volatility
        vol_multiplier = volatility / 0.02  # Normalized to 2% volatility
        adjusted_stop = base_stop_percent * vol_multiplier
        
        # Apply trailing stop if in profit
        if position_type == 'long':
            if current_price > entry_price:
                # Trailing stop
                profit_percent = (current_price - entry_price) / entry_price
                trailing_distance = max(adjusted_stop * 0.5, adjusted_stop - profit_percent * 0.5)
                stop_price = current_price * (1 - trailing_distance)
            else:
                # Fixed stop
                stop_price = entry_price * (1 - adjusted_stop)
        else:  # short position
            if current_price < entry_price:
                # Trailing stop
                profit_percent = (entry_price - current_price) / entry_price
                trailing_distance = max(adjusted_stop * 0.5, adjusted_stop - profit_percent * 0.5)
                stop_price = current_price * (1 + trailing_distance)
            else:
                # Fixed stop
                stop_price = entry_price * (1 + adjusted_stop)
        
        return stop_price
    
    def portfolio_optimization(self, expected_returns, covariance_matrix, risk_free_rate=0.02):
        """Optimize portfolio weights using Modern Portfolio Theory"""
        n_assets = len(expected_returns)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Maximize Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / (portfolio_std + 1e-10)
            return -sharpe
        
        # Optimize
        result = minimize(negative_sharpe, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def risk_parity_allocation(self, covariance_matrix):
        """Risk parity portfolio allocation"""
        n_assets = covariance_matrix.shape[0]
        
        # Objective: Equal risk contribution
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Equal risk contribution target
            target_contrib = portfolio_vol / n_assets
            
            # Minimize squared deviations from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_contribution, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def calculate_risk_metrics(self, portfolio_data):
        """Calculate comprehensive risk metrics"""
        returns = portfolio_data['returns']
        
        metrics = {
            'volatility': np.std(returns) * np.sqrt(252),
            'downside_volatility': np.std(returns[returns < 0]) * np.sqrt(252),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'beta': self.calculate_beta(returns, portfolio_data.get('benchmark_returns')),
            'information_ratio': self.calculate_information_ratio(
                returns, portfolio_data.get('benchmark_returns')
            )
        }
        
        return metrics
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_beta(self, returns, benchmark_returns):
        """Calculate beta relative to benchmark"""
        if benchmark_returns is None or len(benchmark_returns) != len(returns):
            return 1.0
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / (benchmark_variance + 1e-10)
    
    def calculate_information_ratio(self, returns, benchmark_returns):
        """Calculate information ratio"""
        if benchmark_returns is None or len(benchmark_returns) != len(returns):
            return 0
        
        active_returns = returns - benchmark_returns
        
        return np.mean(active_returns) / (np.std(active_returns) + 1e-10) * np.sqrt(252)