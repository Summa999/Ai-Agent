from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class MarketInterface(ABC):
    """Abstract base class for all market interfaces"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connected = False
        self.market_type = "unknown"
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the market"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the market"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Get historical price data"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Dict:
        """Get current bid/ask prices"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, order_type: str, volume: float, 
                         price: float = None, sl: float = None, tp: float = None) -> Dict:
        """Place an order"""
        pass
    
    @abstractmethod
    async def close_position(self, position_id: str) -> bool:
        """Close a specific position"""
        pass
    
    @abstractmethod
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        pass
    
    @abstractmethod
    async def modify_position(self, position_id: str, sl: float = None, tp: float = None) -> bool:
        """Modify an existing position"""
        pass
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        return True  # Override in subclasses