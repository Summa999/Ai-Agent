import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from .market_interface import MarketInterface

class MT5Connector(MarketInterface):
    """MetaTrader 5 connector for forex trading"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.account = config.get('account')
        self.password = config.get('password')
        self.server = config.get('server')
        self.terminal_path = config.get('terminal_path')
        self.symbols = config.get('symbols', ['EURUSD', 'GBPUSD'])
        self.magic_number = config.get('magic_number', 123456)
        
    async def connect(self) -> bool:
        """Connect to MT5 terminal - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._connect_sync)
    
    def _connect_sync(self) -> bool:
        """Synchronous connect method"""
        try:
            # Initialize MT5
            if self.terminal_path:
                if not mt5.initialize(self.terminal_path):
                    self.logger.error("MT5 initialization failed with custom path")
                    return False
            else:
                if not mt5.initialize():
                    self.logger.error("MT5 initialization failed")
                    return False
            
            # Login if credentials provided
            if self.account and self.password and self.server:
                authorized = mt5.login(
                    login=int(self.account),
                    password=self.password,
                    server=self.server
                )
                
                if not authorized:
                    self.logger.error(f"Failed to login to account {self.account}")
                    mt5.shutdown()
                    return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                self.account_info = {
                    'account': account_info.login,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'margin_level': account_info.margin_level,
                    'currency': account_info.currency,
                    'leverage': account_info.leverage,
                    'server': account_info.server
                }
                self.connected = True
                self.logger.info(f"Connected to MT5: Account {account_info.login}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from MT5 - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._disconnect_sync)
    
    def _disconnect_sync(self) -> bool:
        """Synchronous disconnect method"""
        try:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from MT5: {e}")
            return False
    
    async def get_account_info(self) -> Dict:
        """Get account information - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_account_info_sync)
    
    def _get_account_info_sync(self) -> Dict:
        """Synchronous get account info"""
        if not self.connected:
            return {}
            
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    'account': account_info.login,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'margin_level': account_info.margin_level,
                    'currency': account_info.currency,
                    'leverage': account_info.leverage,
                    'profit': account_info.profit
                }
            return self.account_info
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return self.account_info
    
    async def get_historical_data(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Get historical OHLCV data - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_historical_data_sync, symbol, timeframe, bars)
    
    def _get_historical_data_sync(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Synchronous get historical data"""
        try:
            # Convert timeframe string to MT5 timeframe
            tf_dict = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
            
            mt5_timeframe = tf_dict.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Failed to get rates for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'tick_volume']]
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_current_price(self, symbol: str) -> Dict:
        """Get current bid/ask prices - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_current_price_sync, symbol)
    
    def _get_current_price_sync(self, symbol: str) -> Dict:
        """Synchronous get current price"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': tick.ask - tick.bid,
                    'time': datetime.fromtimestamp(tick.time)
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return {}
    
    async def place_order(self, symbol: str, order_type: str, volume: float,
                         price: float, sl: float = None, tp: float = None) -> Dict:
        """Place an order - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._place_order_sync, symbol, order_type, volume, price, sl, tp)
    
    def _place_order_sync(self, symbol: str, order_type: str, volume: float,
                         price: float, sl: float = None, tp: float = None) -> Dict:
        """Synchronous place order"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'retcode': -1, 'comment': f'Symbol {symbol} not found'}
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {'retcode': -1, 'comment': f'Failed to select {symbol}'}
            
            # Prepare request
            point = symbol_info.point
            
            # Determine order type
            if order_type.upper() in ['BUY', 'LONG']:
                mt5_order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            else:
                mt5_order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            
            # Prepare the order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "magic": self.magic_number,
                "comment": "AI Agent Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Add stop loss and take profit if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result:
                return {
                    'retcode': result.retcode,
                    'order': result.order,
                    'volume': result.volume,
                    'price': result.price,
                    'comment': result.comment
                }
            
            return {'retcode': -1, 'comment': 'Order send failed'}
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'retcode': -1, 'comment': str(e)}
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_open_positions_sync)
    
    def _get_open_positions_sync(self) -> List[Dict]:
        """Synchronous get open positions"""
        try:
            positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,  # 0 = buy, 1 = sell
                    'volume': pos.volume,
                    'price': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'time': datetime.fromtimestamp(pos.time)
                })
            
            return position_list
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def close_position(self, position_id: int) -> bool:
        """Close a specific position - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._close_position_sync, position_id)
    
    def _close_position_sync(self, position_id: int) -> bool:
        """Synchronous close position"""
        try:
            position = mt5.positions_get(ticket=position_id)
            
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False
            
            position = position[0]
            
            # Prepare close request
            symbol = position.symbol
            volume = position.volume
            
            # Determine close type (opposite of open)
            if position.type == 0:  # Buy position
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:  # Sell position
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": position_id,
                "price": price,
                "magic": self.magic_number,
                "comment": "AI Agent Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_id} closed successfully")
                return True
            
            self.logger.error(f"Failed to close position {position_id}: {result.comment if result else 'Unknown error'}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    async def modify_position(self, position_id: int, sl: float = None, tp: float = None) -> bool:
        """Modify position stop loss or take profit - ASYNC VERSION"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._modify_position_sync, position_id, sl, tp)
    
    def _modify_position_sync(self, position_id: int, sl: float = None, tp: float = None) -> bool:
        """Synchronous modify position"""
        try:
            position = mt5.positions_get(ticket=position_id)
            
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False
            
            position = position[0]
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position_id,
                "symbol": position.symbol,
                "magic": self.magic_number,
                "comment": "AI Agent Modify"
            }
            
            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            else:
                request["sl"] = position.sl
                
            if tp is not None:
                request["tp"] = tp
            else:
                request["tp"] = position.tp
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {position_id} modified successfully")
                return True
            
            self.logger.error(f"Failed to modify position {position_id}: {result.comment if result else 'Unknown error'}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return False