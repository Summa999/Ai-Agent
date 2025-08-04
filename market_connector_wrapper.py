import asyncio
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class AsyncConnectorWrapper:
    """Wrapper to make synchronous connectors async-compatible"""
    
    def __init__(self, sync_connector):
        self.connector = sync_connector
        self.loop = asyncio.get_event_loop()
    
    async def connect(self) -> bool:
        """Async wrapper for connect"""
        return await self.loop.run_in_executor(None, self.connector.connect)
    
    async def disconnect(self) -> bool:
        """Async wrapper for disconnect"""
        return await self.loop.run_in_executor(None, self.connector.disconnect)
    
    async def get_account_info(self) -> Dict:
        """Async wrapper for get_account_info"""
        return await self.loop.run_in_executor(None, self.connector.get_account_info)
    
    async def get_historical_data(self, symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
        """Async wrapper for get_historical_data"""
        return await self.loop.run_in_executor(
            None, 
            self.connector.get_historical_data,
            symbol, 
            timeframe, 
            bars
        )
    
    async def get_current_price(self, symbol: str) -> Dict:
        """Async wrapper for get_current_price"""
        return await self.loop.run_in_executor(
            None,
            self.connector.get_current_price,
            symbol
        )
    
    async def place_order(self, symbol: str, order_type: str, volume: float,
                         price: float, sl: float = None, tp: float = None) -> Dict:
        """Async wrapper for place_order"""
        return await self.loop.run_in_executor(
            None,
            self.connector.place_order,
            symbol,
            order_type,
            volume,
            price,
            sl,
            tp
        )
    
    async def get_open_positions(self) -> List[Dict]:
        """Async wrapper for get_open_positions"""
        return await self.loop.run_in_executor(None, self.connector.get_open_positions)
    
    async def close_position(self, position_id) -> bool:
        """Async wrapper for close_position"""
        return await self.loop.run_in_executor(
            None,
            self.connector.close_position,
            position_id
        )
    
    async def modify_position(self, position_id, sl: float = None, tp: float = None) -> bool:
        """Async wrapper for modify_position"""
        return await self.loop.run_in_executor(
            None,
            self.connector.modify_position,
            position_id,
            sl,
            tp
        )