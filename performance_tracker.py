import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

class PerformanceTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trades = []
        self.daily_equity = []
        self.positions = {}
        self.metrics_history = []
        self.recent_trades = []
        
    def add_trade(self, trade: Dict):
        """Add completed trade to history"""
        self.trades.append(trade)
        self.recent_trades.append(trade)
        
        # Keep only last 100 recent trades
        if len(self.recent_trades) > 100:
            self.recent_trades = self.recent_trades[-100:]
    
    def update(self, positions: Dict, account_balance: float):
        """Update tracker with current positions and balance"""
        self.positions = positions
        self.daily_equity.append({
            'timestamp': datetime.now(),
            'balance': account_balance,
            'open_positions': len(positions)
        })
        
        # Calculate and store metrics
        if len(self.trades) > 0:
            metrics = self.calculate_metrics()
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        
        # Financial metrics
        total_profit = df['profit'].sum()
        gross_profit = df[df['profit'] > 0]['profit'].sum()
        gross_loss = abs(df[df['profit'] < 0]['profit'].sum())
        
        # Performance ratios
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        returns = self.calculate_returns()
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_drawdown = self.calculate_max_drawdown()
        
        # Trade duration
        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
            avg_duration = df['duration'].mean()
        else:
            avg_duration = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_profit': avg_profit,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_duration_hours': avg_duration
        }
    
    def calculate_returns(self) -> pd.Series:
        """Calculate returns series from equity curve"""
        if len(self.daily_equity) < 2:
            return pd.Series()
        
        equity_df = pd.DataFrame(self.daily_equity)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['balance'].pct_change().dropna()
        return returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 365
        
        if returns.std() == 0:
            return 0.0
        
        return np.sqrt(365) * excess_returns.mean() / returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series,
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 365
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(365) * excess_returns.mean() / downside_returns.std()
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.daily_equity) < 2:
            return 0.0
        
        equity_df = pd.DataFrame(self.daily_equity)
        cumulative = equity_df['balance'].values
        
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown"""
        if len(self.daily_equity) < 2:
            return 0.0
        
        equity_df = pd.DataFrame(self.daily_equity)
        current_balance = equity_df['balance'].iloc[-1]
        peak_balance = equity_df['balance'].max()
        
        if peak_balance > 0:
            return (peak_balance - current_balance) / peak_balance
        
        return 0.0
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        metrics = self.calculate_metrics()
        
        # Add time-based analysis
        if self.trades:
            df = pd.DataFrame(self.trades)
            
            # Daily performance
            if 'exit_time' in df.columns:
                df['date'] = pd.to_datetime(df['exit_time']).dt.date
                daily_pnl = df.groupby('date')['profit'].sum()
                
                metrics['best_day'] = daily_pnl.max()
                metrics['worst_day'] = daily_pnl.min()
                metrics['avg_daily_pnl'] = daily_pnl.mean()
                metrics['daily_win_rate'] = (daily_pnl > 0).sum() / len(daily_pnl)
            
            # By symbol performance
            if 'symbol' in df.columns:
                symbol_performance = df.groupby('symbol').agg({
                    'profit': ['sum', 'count', 'mean'],
                    'symbol': 'size'
                })
                metrics['symbol_performance'] = symbol_performance.to_dict()
            
            # By time of day
            if 'entry_time' in df.columns:
                df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
                hourly_performance = df.groupby('hour')['profit'].mean()
                metrics['best_hours'] = hourly_performance.nlargest(3).to_dict()
                metrics['worst_hours'] = hourly_performance.nsmallest(3).to_dict()
        
        # Streak analysis
        metrics['current_streak'] = self.calculate_current_streak()
        metrics['max_win_streak'] = self.calculate_max_streak('win')
        metrics['max_loss_streak'] = self.calculate_max_streak('loss')
        
        # Risk analysis
        metrics['risk_reward_ratio'] = self.calculate_risk_reward_ratio()
        metrics['recovery_factor'] = self.calculate_recovery_factor(metrics)
        
        return metrics
    
    def calculate_current_streak(self) -> Dict:
        """Calculate current win/loss streak"""
        if not self.trades:
            return {'type': 'none', 'count': 0}
        
        streak_type = 'win' if self.trades[-1]['profit'] > 0 else 'loss'
        count = 1
        
        for trade in reversed(self.trades[:-1]):
            if (trade['profit'] > 0 and streak_type == 'win') or \
               (trade['profit'] < 0 and streak_type == 'loss'):
                count += 1
            else:
                break
        
        return {'type': streak_type, 'count': count}
    
    def calculate_max_streak(self, streak_type: str) -> int:
        """Calculate maximum win or loss streak"""
        if not self.trades:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for trade in self.trades:
            if streak_type == 'win' and trade['profit'] > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            elif streak_type == 'loss' and trade['profit'] < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def calculate_risk_reward_ratio(self) -> float:
        """Calculate average risk/reward ratio"""
        if not self.trades:
            return 0.0
        
        ratios = []
        
        for trade in self.trades:
            if 'stop_loss' in trade and 'take_profit' in trade and 'entry_price' in trade:
                risk = abs(trade['entry_price'] - trade['stop_loss'])
                reward = abs(trade['take_profit'] - trade['entry_price'])
                
                if risk > 0:
                    ratios.append(reward / risk)
        
        return np.mean(ratios) if ratios else 0.0
    
    def calculate_recovery_factor(self, metrics: Dict) -> float:
        """Calculate recovery factor (total profit / max drawdown)"""
        max_dd = metrics.get('max_drawdown', 0)
        total_profit = metrics.get('total_profit', 0)
        
        if max_dd > 0:
            return total_profit / max_dd
        
        return float('inf') if total_profit > 0 else 0.0
    
    def get_recent_trades(self, n: int = 50) -> List[Dict]:
        """Get n most recent trades"""
        return self.recent_trades[-n:]
    
    def add_trade_result(self, trade_result: Dict):
        """Add trade result for model learning"""
        self.recent_trades.append(trade_result)