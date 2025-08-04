import MetaTrader5 as mt5
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

@dataclass
class TradeSignal:
    symbol: str
    direction: int  # 1 for buy, -1 for sell
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    reason: str

class TradeExecutor:
    def __init__(self, mt5_connector, risk_manager):
        self.mt5 = mt5_connector
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        self.active_trades = {}
        self.trade_history = []
        
    def execute_signal(self, signal: TradeSignal, account_info: Dict) -> Optional[Dict]:
        """Execute trade signal"""
        try:
            # Validate signal
            if not self.validate_signal(signal, account_info):
                return None
            
            # Check risk limits
            if not self.risk_manager.check_trade_allowed(signal, account_info):
                self.logger.warning(f"Trade not allowed due to risk limits: {signal.symbol}")
                return None
            
            # Determine order type
            order_type = mt5.ORDER_TYPE_BUY if signal.direction == 1 else mt5.ORDER_TYPE_SELL
            
            # Execute trade
            result = self.mt5.place_order(
                symbol=signal.symbol,
                order_type=order_type,
                volume=signal.position_size,
                price=signal.entry_price,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"AI_{signal.confidence:.2f}_{signal.reason}"
            )
            
            if result:
                # Record trade
                trade_record = {
                    'ticket': result['ticket'],
                    'signal': signal,
                    'entry_time': result['time'],
                    'entry_price': result['price'],
                    'status': 'active'
                }
                
                self.active_trades[result['ticket']] = trade_record
                self.logger.info(f"Trade executed: {signal.symbol} {order_type} @ {result['price']}")
                
                return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            
        return None
    
    def validate_signal(self, signal: TradeSignal, account_info: Dict) -> bool:
        """Validate trade signal"""
        # Check confidence threshold
        if signal.confidence < 0.6:
            self.logger.debug(f"Signal confidence too low: {signal.confidence}")
            return False
        
        # Check position size
        if signal.position_size <= 0:
            self.logger.error("Invalid position size")
            return False
        
        # Check stop loss and take profit
        if signal.direction == 1:  # Buy
            if signal.stop_loss >= signal.entry_price:
                self.logger.error("Invalid stop loss for buy signal")
                return False
            if signal.take_profit <= signal.entry_price:
                self.logger.error("Invalid take profit for buy signal")
                return False
        else:  # Sell
            if signal.stop_loss <= signal.entry_price:
                self.logger.error("Invalid stop loss for sell signal")
                return False
            if signal.take_profit >= signal.entry_price:
                self.logger.error("Invalid take profit for sell signal")
                return False
        
        # Check if we already have a position in this symbol
        if self.has_open_position(signal.symbol):
            self.logger.debug(f"Already have open position in {signal.symbol}")
            return False
        
        return True
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol"""
        positions = mt5.positions_get(symbol=symbol)
        return positions is not None and len(positions) > 0
    
    def manage_positions(self):
        """Manage all open positions"""
        positions = mt5.positions_get()
        
        if positions is None:
            return
        
        for position in positions:
            if position.ticket in self.active_trades:
                self.manage_single_position(position)
    
    def manage_single_position(self, position):
        """Manage a single position"""
        trade_record = self.active_trades.get(position.ticket)
        if not trade_record:
            return
        
        signal = trade_record['signal']
        
        # Calculate current profit percentage
        if position.type == 0:  # Buy
            profit_pct = (position.price_current - position.price_open) / position.price_open
        else:  # Sell
            profit_pct = (position.price_open - position.price_current) / position.price_open
        
        # Trailing stop logic
        if profit_pct > 0.02:  # 2% profit
            new_sl = self.calculate_trailing_stop(position, signal)
            if new_sl:
                self.mt5.modify_position(position.ticket, sl=new_sl)
        
        # Check if position should be closed
        if self.should_close_position(position, signal):
            self.close_position(position.ticket)
    
    def calculate_trailing_stop(self, position, signal: TradeSignal) -> Optional[float]:
        """Calculate trailing stop price"""
        current_price = position.price_current
        entry_price = position.price_open
        
        if position.type == 0:  # Buy
            # Move stop to breakeven after 2% profit
            profit = (current_price - entry_price) / entry_price
            if profit > 0.02 and position.sl < entry_price:
                return entry_price + (entry_price * 0.001)  # Small buffer above entry
            
            # Trail stop at 50% of profit
            if profit > 0.03:
                trail_distance = (current_price - entry_price) * 0.5
                new_sl = entry_price + trail_distance
                if new_sl > position.sl:
                    return new_sl
                    
        else:  # Sell
            profit = (entry_price - current_price) / entry_price
            if profit > 0.02 and position.sl > entry_price:
                return entry_price - (entry_price * 0.001)
            
            if profit > 0.03:
                trail_distance = (entry_price - current_price) * 0.5
                new_sl = entry_price - trail_distance
                if new_sl < position.sl:
                    return new_sl
        
        return None
    
    def should_close_position(self, position, signal: TradeSignal) -> bool:
        """Determine if position should be closed"""
        # Time-based exit
        hold_time = (datetime.now() - self.active_trades[position.ticket]['entry_time']).total_seconds() / 3600
        if hold_time > 24:  # Close after 24 hours
            return True
        
        # Reverse signal
        # This would need access to current signals
        
        return False
    
    def close_position(self, ticket: int) -> bool:
        """Close a position and record results"""
        result = self.mt5.close_position(ticket)
        
        if result:
            # Get final position details
            position = mt5.history_orders_get(ticket=ticket)
            if position:
                position = position[0]
                
                # Calculate profit
                trade_record = self.active_trades.get(ticket)
                if trade_record:
                    profit = position.profit
                    
                    # Update trade record
                    trade_record['exit_time'] = datetime.now()
                    trade_record['exit_price'] = position.price_current
                    trade_record['profit'] = profit
                    trade_record['status'] = 'closed'
                    
                    # Move to history
                    self.trade_history.append(trade_record)
                    del self.active_trades[ticket]
                    
                    self.logger.info(f"Position {ticket} closed. Profit: {profit}")
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        if not self.trade_history:
            return {}
        
        profits = [trade['profit'] for trade in self.trade_history]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        stats = {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            'total_profit': sum(profits),
            'average_profit': np.mean(profits) if profits else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0,
            'max_drawdown': self.calculate_max_drawdown(profits)
        }
        
        return stats
    
    def calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown from profit list"""
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        return float(np.min(drawdown))