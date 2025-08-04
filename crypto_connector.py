import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from .market_interface import MarketInterface

class CryptoConnector(MarketInterface):
    """Connector for cryptocurrency exchanges using CCXT (Sync version)"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.exchange_name = config.get('exchange', 'binance')
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', False)
        self.symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        self.exchange = None
        self.markets = {}
        self.connected = False
        self.account_info = {}
        
    async def connect(self) -> bool:
        """Connect to crypto exchange (using sync methods)"""
        try:
            # Exchange configuration
            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                },
                'timeout': 30000,
                'rateLimit': 1200,
            }
            
            # Add API credentials if provided
            if self.api_key and self.api_secret:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
                self.logger.info(f"Using API credentials for {self.exchange_name}")
            else:
                self.logger.info(f"No API credentials - connecting in read-only mode")
            
            # Use testnet if specified
            if self.testnet:
                if self.exchange_name == 'binance':
                    exchange_config['sandbox'] = True
                    self.logger.info("Using Binance testnet")
            
            # Create exchange object with warning suppression
            self.exchange = ccxt.binance(exchange_config)
            
            # Suppress the fetchOpenOrders warning
            if hasattr(self.exchange, 'options'):
                self.exchange.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
            
            self.logger.info(f"Exchange object created for {self.exchange_name}")
            
            # Load markets using sync method (no await)
            try:
                self.markets = self.exchange.load_markets()
                self.logger.info(f"Loaded {len(self.markets)} markets from {self.exchange_name}")
            except Exception as market_error:
                self.logger.error(f"Failed to load markets: {market_error}")
                return False
            
            # Try authenticated connection if API keys are provided
            if self.api_key and self.api_secret:
                try:
                    # Test connection with balance query (sync)
                    balance = self.exchange.fetch_balance()
                    
                    usdt_balance = balance.get('USDT', {}).get('free', 0) if balance.get('USDT') else 0
                    total_balance = balance.get('total', {}).get('USDT', usdt_balance) if balance.get('total') else usdt_balance
                    
                    self.account_info = {
                        'account': 'live_account',
                        'balance': float(total_balance),
                        'currency': 'USDT',
                        'type': 'authenticated'
                    }
                    
                    self.connected = True
                    self.logger.info(f"✓ Connected to {self.exchange_name} with authentication - Balance: ${total_balance:.2f}")
                    return True
                    
                except Exception as auth_error:
                    self.logger.warning(f"Authentication failed: {auth_error}")
                    self.logger.info("Switching to read-only mode...")
            
            # Read-only mode
            self.connected = True
            self.account_info = {
                'account': 'demo_account',
                'balance': 10000.0,
                'currency': 'USDT',
                'type': 'read_only'
            }
            
            self.logger.info(f"✓ Connected to {self.exchange_name} in read-only mode - Demo Balance: $10,000")
            return True
                
        except Exception as e:
            self.logger.error(f"✗ Failed to connect to {self.exchange_name}: {e}")
            self.connected = False
            return False
    
    async def get_historical_data(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Get historical OHLCV data (sync method wrapped in async)"""
        try:
            if not self.connected or not self.exchange:
                return pd.DataFrame()
            
            # Convert timeframe
            tf_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
            }
            ccxt_timeframe = tf_map.get(timeframe, '1h')
            
            # Calculate since time
            if ccxt_timeframe.endswith('m'):
                minutes = int(ccxt_timeframe[:-1])
                start_time = datetime.now() - timedelta(minutes=minutes * bars)
            elif ccxt_timeframe.endswith('h'):
                hours = int(ccxt_timeframe[:-1])
                start_time = datetime.now() - timedelta(hours=hours * bars)
            else:
                start_time = datetime.now() - timedelta(hours=bars)
            
            since = int(start_time.timestamp() * 1000)
            
            # Use sync method (no await)
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, ccxt_timeframe, since, bars
            )
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    async def get_current_price(self, symbol: str) -> Dict:
        """Get current ticker data (sync method wrapped)"""
        try:
            if not self.connected or not self.exchange:
                return {}
            
            # Use sync method
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'last': ticker.get('last', 0),
                'volume': ticker.get('baseVolume', 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return {}
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        return self.account_info if self.connected else {}
    
    async def disconnect(self) -> bool:
        """Disconnect from exchange"""
        try:
            if self.exchange:
                # Note: ccxt 4.4.98 doesn't have close() method
                pass  # Just set connected to False
            self.connected = False
            self.exchange = None
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")
            return False
    
    async def place_order(self, symbol: str, order_type: str, volume: float,
                         price: float = None, sl: float = None, tp: float = None) -> Dict:
        """Place an order (demo mode if no API key)"""
        
        if not self.api_key or self.account_info.get('type') == 'read_only':
            self.logger.info(f"DEMO ORDER: {order_type} {volume} {symbol} @ {price}")
            return {
                'retcode': 10009,
                'order': f'demo_{int(datetime.now().timestamp())}',
                'volume': volume,
                'price': price,
                'comment': 'Demo order executed'
            }
        
        try:
            side = 'buy' if order_type.upper() in ['BUY', 'LONG'] else 'sell'
            
            # Use sync method
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=volume,
                price=price
            )
            
            return {
                'retcode': 10009,
                'order': order['id'],
                'volume': order['amount'],
                'price': order['price'],
                'comment': 'Order placed'
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'retcode': -1, 'comment': str(e)}
    
    async def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        if not self.api_key or self.account_info.get('type') == 'read_only':
            return []
        
        try:
            # Use sync method
            orders = self.exchange.fetch_open_orders()
            
            positions = []
            for order in orders:
                positions.append({
                    'ticket': order['id'],
                    'symbol': order['symbol'],
                    'type': 0 if order['side'] == 'buy' else 1,
                    'volume': order['amount'],
                    'price': order['price'],
                    'sl': 0,
                    'tp': 0,
                    'profit': 0,
                    'time': datetime.fromtimestamp(order['timestamp'] / 1000)
                })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def close_position(self, position_id: str) -> bool:
        """Cancel an open order"""
        if not self.api_key or self.account_info.get('type') == 'read_only':
            self.logger.info(f"DEMO: Position {position_id} closed")
            return True
        
        try:
            # Use sync method
            self.exchange.cancel_order(position_id)
            return True
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    async def modify_position(self, position_id: str, sl: float = None, tp: float = None) -> bool:
        """Modify position (not supported in spot trading)"""
        self.logger.warning("Spot trading doesn't support direct position modification")
        return False