import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import logging

class PortfolioOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def optimize_portfolio(self, returns_data: pd.DataFrame,
                         constraints: Dict = None) -> Dict:
        """Optimize portfolio allocation"""
        # Calculate expected returns and covariance
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        n_assets = len(expected_returns)
        
        # Optimization constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 0.4,
                'target_return': None,
                'max_volatility': None
            }
        
        # Define optimization problem
        def portfolio_stats(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_vol
        
        # Objective function (maximize Sharpe ratio)
        def neg_sharpe(weights):
            p_ret, p_vol = portfolio_stats(weights)
            return -(p_ret / p_vol) * np.sqrt(252)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if constraints['target_return']:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: portfolio_stats(x)[0] - constraints['target_return']
            })
        
        if constraints['max_volatility']:
            cons.append({
                'type': 'ineq',
                'fun': lambda x: constraints['max_volatility'] - portfolio_stats(x)[1]
            })
        
        # Bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        optimal_weights = result.x
        opt_return, opt_vol = portfolio_stats(optimal_weights)
        opt_sharpe = (opt_return / opt_vol) * np.sqrt(252)
        
        # Calculate other portfolio types
        min_vol_weights = self.minimum_volatility_portfolio(cov_matrix, constraints)
        max_return_weights = self.maximum_return_portfolio(expected_returns, constraints)
        risk_parity_weights = self.risk_parity_portfolio(cov_matrix)
        
        return {
            'optimal_weights': dict(zip(returns_data.columns, optimal_weights)),
            'expected_return': opt_return * 252,
            'expected_volatility': opt_vol * np.sqrt(252),
            'sharpe_ratio': opt_sharpe,
            'min_volatility_weights': dict(zip(returns_data.columns, min_vol_weights)),
            'max_return_weights': dict(zip(returns_data.columns, max_return_weights)),
            'risk_parity_weights': dict(zip(returns_data.columns, risk_parity_weights))
        }
    
    def minimum_volatility_portfolio(self, cov_matrix: np.ndarray,
                                   constraints: Dict) -> np.ndarray:
        """Find minimum volatility portfolio"""
        n = cov_matrix.shape[0]
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(n))
        x0 = np.array([1/n] * n)
        
        result = minimize(portfolio_vol, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x
    
    def maximum_return_portfolio(self, expected_returns: pd.Series,
                               constraints: Dict) -> np.ndarray:
        """Find maximum return portfolio"""
        n = len(expected_returns)
        
        def neg_returns(weights):
            return -np.dot(weights, expected_returns)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(n))
        x0 = np.array([1/n] * n)
        
        result = minimize(neg_returns, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x
    
    def risk_parity_portfolio(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk parity portfolio"""
        n = cov_matrix.shape[0]
        
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Equal risk contribution
            target_contrib = portfolio_vol / n
            
            return np.sum((contrib - target_contrib) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.array([1/n] * n)
        
        result = minimize(risk_budget_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x
    
    def calculate_efficient_frontier(self, returns_data: pd.DataFrame,
                                   n_portfolios: int = 100) -> pd.DataFrame:
        """Calculate efficient frontier"""
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        # Range of target returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        frontier_weights = []
        frontier_returns = []
        frontier_volatility = []
        
        for target_return in target_returns:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'target_return': target_return,
                'max_volatility': None
            }
            
            # Find minimum volatility for this return
            weights = self.minimum_volatility_portfolio(cov_matrix, constraints)
            
            # Calculate portfolio stats
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            frontier_weights.append(weights)
            frontier_returns.append(portfolio_return)
            frontier_volatility.append(portfolio_vol)
        
        return pd.DataFrame({
            'returns': np.array(frontier_returns) * 252,
            'volatility': np.array(frontier_volatility) * np.sqrt(252),
            'sharpe': (np.array(frontier_returns) / np.array(frontier_volatility)) * np.sqrt(252)
        })