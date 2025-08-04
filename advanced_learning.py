import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import logging
from decimal import Decimal
from .market_interface import MarketInterface

class CryptoConnector(MarketInterface):
    """
    Comprehensive cryptocurrency exchange connector using CCXT.
    Supports both authenticated and public access to multiple exchanges.
    
    Features:
    - Real-time and historical market data
    - Order management (spot trading)
    - Account information
    - Demo mode when no API keys are provided
    - Automatic rate limiting
    - Support for multiple timeframes
    """
    
    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M'
    }
    
    ORDER_TYPE_MAP = {
        'market': 'market',
        'limit': 'limit',
        'stop': 'stop',
        'stop_limit': 'stop_limit'
    }
    
    def __init__(self, config: Dict):
        """
        Initialize the crypto connector.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - exchange (str): Exchange name (e.g., 'binance', 'ftx')
                - api_key (str, optional): API key for authenticated access
                - api_secret (str, optional): API secret for authenticated access
                - symbols (List[str]): List of trading symbols to monitor
                - testnet (bool): Whether to use testnet/sandbox environment
                - enable_rate_limit (bool): Enable built-in rate limiting
                - verbose (bool): Enable verbose logging from CCXT
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.exchange_name = config.get('exchange', 'binance').lower()
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', False)
        self.symbols = config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        self.enable_rate_limit = config.get('enable_rate_limit', True)
        self.verbose = config.get('verbose', False)
        self.exchange = None
        self.markets = {}
        self.account_info = {}
        self.last_update = None
        
    async def connect(self) -> bool:
        """Connect to the cryptocurrency exchange"""
        try:
            # Exchange configuration
            exchange_config = {
                'enableRateLimit': self.enable_rate_limit,
                'verbose': self.verbose,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            }
            
            # Add API credentials if provided
            if self.api_key and self.api_secret:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
            
            # Configure testnet if specified
            if self.testnet:
                if self.exchange_name == 'binance':
                    exchange_config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api/v3',
                            'private': 'https://testnet.binance.vision/api/v3',
                        }
                    }
                elif self.exchange_name == 'ftx':
                    exchange_config['urls'] = {
                        'api': 'https://ftx.com/api'
                    }
            
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class(exchange_config)
            
            # Load markets
            await self._load_markets()
            
            # For public data only (no authentication needed)
            if not self.api_key:
                self.logger.info(f"Connected to {self.exchange_name} in read-only mode")
                self.connected = True
                self._set_demo_account_info()
                return True
            
            # For authenticated connection
            try:
                # Test connection with balance query
                await self._update_account_info()
                self.connected = True
                self.logger.info(f"Successfully connected to {self.exchange_name}")
                return True
                
            except Exception as auth_error:
                self.logger.warning(f"Authentication failed, switching to read-only mode: {auth_error}")
                self._set_demo_account_info()
                self.connected = True
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.exchange_name}: {e}", exc_info=True)
            self.connected = False
            return False
    
    async def _load_markets(self) -> None:
        """Load and cache market information"""
        try:
            self.markets = await self.exchange.load_markets()
            self.logger.debug(f"Loaded {len(self.markets)} markets")
        except Exception as e:
            self.logger.error(f"Error loading markets: {e}")
            self.markets = {}
    
    def _set_demo_account_info(self) -> None:
        """Set demo account information for read-only mode"""
        self.account_info = {
            'account_id': 'demo_account',
            'balances': {
                'USDT': {
                    'free': 10000,
                    'used': 0,
                    'total': 10000
                },
                'BTC': {
                    'free': 0.5,
                    'used': 0,
                    'total': 0.5
                }
            },
            'permissions': ['read'],
            'timestamp': datetime.now()
        }
    
    async def _update_account_info(self) -> None:
        """Update account information from exchange"""
        try:
            balance = await self.exchange.fetch_balance()
            self.account_info = {
                'account_id': balance.get('info', {}).get('accountId', 'live_account'),
                'balances': {k: v for k, v in balance['total'].items() if v > 0},
                'permissions': balance.get('info', {}).get('permissions', ['read']),
                'timestamp': datetime.now()
            }
            self.last_update = datetime.now()
        except Exception as e:
            self.logger.error(f"Error updating account info: {e}")
            raise
    
    async def disconnect(self) -> bool:
        """Disconnect from exchange"""
        try:
            if self.exchange:
                await self.exchange.close()
            self.connected = False
            self.logger.info(f"Disconnected from {self.exchange_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting: {e}")
            return False
    
    async def get_account_info(self) -> Dict:
        """Get current account information"""
        if not self.connected:
            return {}
        
        # Update account info if authenticated and data is stale (>1 minute old)
        if self.api_key and (not self.last_update or (datetime.now() - self.last_update) > timedelta(minutes=1)):
            try:
                await self._update_account_info()
            except Exception as e:
                self.logger.warning(f"Couldn't update account info: {e}")
        
        return self.account_info
    
    async def get_balance(self, currency: str = 'USDT') -> Dict[str, float]:
        """Get balance for a specific currency"""
        account_info = await self.get_account_info()
        return account_info.get('balances', {}).get(currency, {'free': 0, 'used': 0, 'total': 0})
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        since: Optional[Union[datetime, int]] = None,
        limit: int = 500,
        params: Dict = {}
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            since: Optional start time (datetime or timestamp in ms)
            limit: Number of candles to fetch
            params: Additional parameters for the exchange
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data and timestamp index
        """
        try:
            # Validate symbol
            if symbol not in self.symbols and symbol not in self.markets:
                raise ValueError(f"Symbol {symbol} not available")
            
            # Convert timeframe format
            ccxt_timeframe = self.TIMEFRAME_MAP.get(timeframe, '1h')
            
            # Calculate since if not provided
            if since is None:
                if ccxt_timeframe.endswith('m'):
                    minutes = int(ccxt_timeframe[:-1])
                    since = datetime.now() - timedelta(minutes=minutes * limit)
                elif ccxt_timeframe.endswith('h'):
                    hours = int(ccxt_timeframe[:-1])
                    since = datetime.now() - timedelta(hours=hours * limit)
                elif ccxt_timeframe.endswith('d'):
                    days = int(ccxt_timeframe[:-1])
                    since = datetime.now() - timedelta(days=days * limit)
                else:
                    since = datetime.now() - timedelta(hours=limit)
            
            # Convert datetime to timestamp in milliseconds
            if isinstance(since, datetime):
                since = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=ccxt_timeframe,
                since=since,
                limit=limit,
                params=params
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def get_current_price(self, symbol: str) -> Dict:
        """Get current ticker data"""
        try:
            if symbol not in self.symbols and symbol not in self.markets:
                raise ValueError(f"Symbol {symbol} not available")
            
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'bid': float(ticker.get('bid', 0)),
                'ask': float(ticker.get('ask', 0)),
                'last': float(ticker.get('last', 0)),
                'bid_volume': float(ticker.get('bidVolume', 0)),
                'ask_volume': float(ticker.get('askVolume', 0)),
                'base_volume': float(ticker.get('baseVolume', 0)),
                'quote_volume': float(ticker.get('quoteVolume', 0)),
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000) if 'timestamp' in ticker else datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return {
                'symbol': symbol,
                'bid': 0,
                'ask': 0,
                'last': 0,
                'timestamp': datetime.now()
            }
    
    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """Get order book for a symbol"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=limit)
            
            return {
                'symbol': symbol,
                'bids': [[float(price), float(amount)] for price, amount in orderbook['bids']],
                'asks': [[float(price), float(amount)] for price, amount in orderbook['asks']],
                'timestamp': datetime.fromtimestamp(orderbook['timestamp'] / 1000) if 'timestamp' in orderbook else datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return {
                'symbol': symbol,
                'bids': [],
                'asks': [],
                'timestamp': datetime.now()
            }
    
    async def place_order(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        amount: float, 
        price: Optional[float] = None,
        params: Dict = {}
    ) -> Dict:
        """
        Place an order on the exchange
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', etc.
            amount: Amount to buy/sell
            price: Price for limit orders
            params: Additional order parameters
            
        Returns:
            Dict: Order information
        """
        # Check if we have API credentials
        if not self.api_key:
            # Demo mode - simulate order
            order_id = f'demo_{datetime.now().timestamp()}'
            self.logger.info(f"DEMO ORDER: {side} {amount} {symbol} @ {price if price else 'market'}")
            return {
                'id': order_id,
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price if price else await self.get_current_price(symbol)['last'],
                'status': 'closed',
                'filled': amount,
                'timestamp': datetime.now(),
                'info': {'demo_order': True}
            }
        
        try:
            # Validate inputs
            if symbol not in self.symbols and symbol not in self.markets:
                raise ValueError(f"Symbol {symbol} not available")
            
            if side.lower() not in ['buy', 'sell']:
                raise ValueError("Side must be 'buy' or 'sell'")
            
            order_type = self.ORDER_TYPE_MAP.get(order_type.lower(), 'limit')
            
            # For market orders, ensure we have enough balance
            if order_type == 'market':
                balance = await self.get_balance('USDT' if side == 'buy' else symbol.split('/')[0])
                if balance['free'] < (amount * (price if price else 1)):
                    raise ValueError("Insufficient balance")
            
            # Place the order
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            # Format the response
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': float(order['amount']),
                'price': float(order['price']) if 'price' in order else None,
                'status': order['status'],
                'filled': float(order['filled']) if 'filled' in order else 0,
                'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000) if 'timestamp' in order else datetime.now(),
                'info': order.get('info', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}", exc_info=True)
            return {
                'error': str(e),
                'status': 'rejected'
            }
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """Get order information by order ID"""
        if not self.api_key:
            return {
                'id': order_id,
                'status': 'closed',
                'info': {'demo_order': True}
            }
        
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return {
                'id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': float(order['amount']),
                'price': float(order['price']) if 'price' in order else None,
                'status': order['status'],
                'filled': float(order['filled']) if 'filled' in order else 0,
                'remaining': float(order['remaining']) if 'remaining' in order else 0,
                'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000) if 'timestamp' in order else datetime.now(),
                'info': order.get('info', {})
            }
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {e}")
            return {
                'error': str(e),
                'status': 'unknown'
            }
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an order"""
        if not self.api_key:
            self.logger.info(f"DEMO: Order {order_id} cancelled")
            return True
        
        try:
            response = await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        if not self.api_key:
            return []
        
        try:
            orders = await self.exchange.fetch_open_orders(symbol) if symbol else await self.exchange.fetch_open_orders()
            
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'amount': float(order['amount']),
                    'price': float(order['price']) if 'price' in order else None,
                    'status': order['status'],
                    'filled': float(order['filled']) if 'filled' in order else 0,
                    'remaining': float(order['remaining']) if 'remaining' in order else 0,
                    'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000) if 'timestamp' in order else datetime.now(),
                    'info': order.get('info', {})
                })
            
            return formatted_orders
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []
    
    async def get_closed_orders(self, symbol: Optional[str] = None, since: Optional[datetime] = None) -> List[Dict]:
        """Get closed orders"""
        if not self.api_key:
            return []
        
        try:
            params = {}
            if since:
                params['since'] = int(since.timestamp() * 1000)
            
            orders = await self.exchange.fetch_closed_orders(symbol, params=params) if symbol else await self.exchange.fetch_closed_orders(params=params)
            
            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    'id': order['id'],
                    'symbol': order['symbol'],
                    'type': order['type'],
                    'side': order['side'],
                    'amount': float(order['amount']),
                    'price': float(order['price']) if 'price' in order else None,
                    'status': order['status'],
                    'filled': float(order['filled']) if 'filled' in order else 0,
                    'remaining': float(order['remaining']) if 'remaining' in order else 0,
                    'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000) if 'timestamp' in order else datetime.now(),
                    'info': order.get('info', {})
                })
            
            return formatted_orders
        except Exception as e:
            self.logger.error(f"Error fetching closed orders: {e}")
            return []
    
    async def get_open_positions(self) -> List[Dict]:
        """Get open positions (for spot trading, these are balances)"""
        if not self.api_key:
            return []
        
        try:
            # For spot trading, we return non-zero balances
            account_info = await self.get_account_info()
            positions = []
            
            for currency, balance in account_info.get('balances', {}).items():
                if balance['total'] > 0:
                    positions.append({
                        'symbol': currency,
                        'amount': balance['total'],
                        'free': balance['free'],
                        'used': balance['used'],
                        'timestamp': account_info.get('timestamp', datetime.now())
                    })
            
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict:
        """Get trading fees for the account or specific symbol"""
        try:
            if not self.api_key:
                return {
                    'maker': 0.001,
                    'taker': 0.001,
                    'info': {'demo_fees': True}
                }
            
            fees = await self.exchange.fetch_trading_fees() if not symbol else await self.exchange.fetch_trading_fee(symbol)
            
            return {
                'maker': float(fees.get('maker', 0)),
                'taker': float(fees.get('taker', 0)),
                'info': fees.get('info', {})
            }
        except Exception as e:
            self.logger.error(f"Error fetching trading fees: {e}")
            return {
                'maker': 0.002,
                'taker': 0.002,
                'info': {'error': str(e)}
            }
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get information about a trading symbol"""
        try:
            if symbol not in self.markets:
                await self._load_markets()
            
            market = self.markets.get(symbol, {})
            
            return {
                'symbol': symbol,
                'base': market.get('base', ''),
                'quote': market.get('quote', ''),
                'precision': {
                    'amount': market.get('precision', {}).get('amount', 0),
                    'price': market.get('precision', {}).get('price', 0)
                },
                'limits': {
                    'amount': {
                        'min': float(market.get('limits', {}).get('amount', {}).get('min', 0)),
                        'max': float(market.get('limits', {}).get('amount', {}).get('max', 0))
                    },
                    'price': {
                        'min': float(market.get('limits', {}).get('price', {}).get('min', 0)),
                        'max': float(market.get('limits', {}).get('price', {}).get('max', 0))
                    },
                    'cost': {
                        'min': float(market.get('limits', {}).get('cost', {}).get('min', 0)),
                        'max': float(market.get('limits', {}).get('cost', {}).get('max', 0))
                    }
                },
                'active': market.get('active', False),
                'info': market.get('info', {})
            }
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[List[Union[float, int]]]:
        """Get raw OHLCV data"""
        try:
            ccxt_timeframe = self.TIMEFRAME_MAP.get(timeframe, '1h')
            
            if since:
                since = int(since.timestamp() * 1000)
            
            return await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=ccxt_timeframe,
                since=since,
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return []
    
    async def get_tickers(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Get tickers for multiple symbols"""
        try:
            symbols = symbols or self.symbols
            tickers = await self.exchange.fetch_tickers(symbols)
            
            formatted_tickers = {}
            for symbol, ticker in tickers.items():
                formatted_tickers[symbol] = {
                    'bid': float(ticker.get('bid', 0)),
                    'ask': float(ticker.get('ask', 0)),
                    'last': float(ticker.get('last', 0)),
                    'high': float(ticker.get('high', 0)),
                    'low': float(ticker.get('low', 0)),
                    'volume': float(ticker.get('baseVolume', 0)),
                    'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000) if 'timestamp' in ticker else datetime.now()
                }
            
            return formatted_tickers
        except Exception as e:
            self.logger.error(f"Error fetching tickers: {e}")
            return {}
    
    async def get_recent_trades(self, symbol: str, since: Optional[datetime] = None, limit: int = 50) -> List[Dict]:
        """Get recent trades for a symbol"""
        try:
            params = {}
            if since:
                params['since'] = int(since.timestamp() * 1000)
            
            trades = await self.exchange.fetch_trades(symbol, since=since, limit=limit, params=params)
            
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    'id': trade['id'],
                    'symbol': trade['symbol'],
                    'side': trade['side'],
                    'price': float(trade['price']),
                    'amount': float(trade['amount']),
                    'cost': float(trade['cost']) if 'cost' in trade else float(trade['price']) * float(trade['amount']),
                    'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000),
                    'info': trade.get('info', {})
                })
            
            return formatted_trades
        except Exception as e:
            self.logger.error(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate for perpetual contracts"""
        try:
            if not hasattr(self.exchange, 'fetchFundingRate'):
                raise NotImplementedError("This exchange doesn't support funding rates")
            
            rate = await self.exchange.fetchFundingRate(symbol)
            
            return {
                'symbol': symbol,
                'rate': float(rate['fundingRate']),
                'timestamp': datetime.fromtimestamp(rate['timestamp'] / 1000),
                'next_funding': datetime.fromtimestamp(rate['nextFundingTime'] / 1000) if 'nextFundingTime' in rate else None,
                'info': rate.get('info', {})
            }
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get klines/candlestick data with more flexible time ranges
        
        Args:
            symbol: Trading symbol
            interval: Time interval (e.g., '1h', '4h', '1d')
            start_time: Optional start time
            end_time: Optional end time
            limit: Maximum number of candles to return
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Convert interval to CCXT timeframe
            timeframe = self.TIMEFRAME_MAP.get(interval, interval)
            
            # Calculate since if start_time is not provided
            since = None
            if start_time:
                since = int(start_time.timestamp() * 1000)
            elif end_time:
                # Estimate start time based on interval and limit
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    start_time = end_time - timedelta(minutes=minutes * limit)
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    start_time = end_time - timedelta(hours=hours * limit)
                elif timeframe.endswith('d'):
                    days = int(timeframe[:-1])
                    start_time = end_time - timedelta(days=days * limit)
                else:
                    start_time = end_time - timedelta(hours=limit)
                
                since = int(start_time.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Filter by end_time if provided
            if end_time:
                df = df[df.index <= end_time]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting klines for {symbol}: {e}")
            return pd.DataFrame()