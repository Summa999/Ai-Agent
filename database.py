import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

class DatabaseManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['name'],
                user=self.config['user'],
                password=self.config['password']
            )
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            self.logger.info("Connected to database")
            
            # Create tables if not exist
            self.create_tables()
            
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
    
    def create_tables(self):
        """Create necessary tables"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DECIMAL(10, 5) NOT NULL,
                high DECIMAL(10, 5) NOT NULL,
                low DECIMAL(10, 5) NOT NULL,
                close DECIMAL(10, 5) NOT NULL,
                volume BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                ticket BIGINT UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price DECIMAL(10, 5) NOT NULL,
                exit_time TIMESTAMP,
                exit_price DECIMAL(10, 5),
                volume DECIMAL(10, 2) NOT NULL,
                stop_loss DECIMAL(10, 5),
                take_profit DECIMAL(10, 5),
                profit DECIMAL(10, 2),
                commission DECIMAL(10, 2),
                swap DECIMAL(10, 2),
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                model_name VARCHAR(50) NOT NULL,
                prediction INTEGER NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                probabilities JSONB,
                features JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                total_profit DECIMAL(10, 2) NOT NULL,
                max_drawdown DECIMAL(5, 4) NOT NULL,
                sharpe_ratio DECIMAL(5, 2),
                win_rate DECIMAL(5, 4),
                profit_factor DECIMAL(5, 2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(50) NOT NULL,
                evaluation_date DATE NOT NULL,
                accuracy DECIMAL(5, 4),
                precision DECIMAL(5, 4),
                recall DECIMAL(5, 4),
                f1_score DECIMAL(5, 4),
                sharpe_ratio DECIMAL(5, 2),
                total_return DECIMAL(10, 4),
                max_drawdown DECIMAL(5, 4),
                parameters JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table in tables:
            try:
                self.cursor.execute(table)
                self.connection.commit()
            except Exception as e:
                self.logger.error(f"Error creating table: {e}")
                self.connection.rollback()
    
    def save_market_data(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Save market data to database"""
        try:
            for idx, row in data.iterrows():
                self.cursor.execute(
                    """
                    INSERT INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """,
                    (symbol, timeframe, idx, row['open'], row['high'], 
                     row['low'], row['close'], row['volume'])
                )
            
            self.connection.commit()
            self.logger.info(f"Saved {len(data)} records for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
            self.connection.rollback()
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve market data from database"""
        try:
            self.cursor.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = %s AND timeframe = %s 
                AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
                """,
                (symbol, timeframe, start_date, end_date)
            )
            
            rows = self.cursor.fetchall()
            
            if rows:
                df = pd.DataFrame(rows)
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        try:
            self.cursor.execute(
                """
                INSERT INTO trades
                (ticket, symbol, direction, entry_time, entry_price, 
                 volume, stop_loss, take_profit, comment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (trade_data['ticket'], trade_data['symbol'], 
                 trade_data['direction'], trade_data['entry_time'],
                 trade_data['entry_price'], trade_data['volume'],
                 trade_data.get('stop_loss'), trade_data.get('take_profit'),
                 trade_data.get('comment', ''))
            )
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {e}")
            self.connection.rollback()
    
    def update_trade(self, ticket: int, exit_data: Dict):
        """Update trade with exit information"""
        try:
            self.cursor.execute(
                """
                UPDATE trades
                SET exit_time = %s, exit_price = %s, profit = %s,
                    commission = %s, swap = %s
                WHERE ticket = %s
                """,
                (exit_data['exit_time'], exit_data['exit_price'],
                 exit_data['profit'], exit_data.get('commission', 0),
                 exit_data.get('swap', 0), ticket)
            )
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            self.connection.rollback()
    
    def save_prediction(self, prediction_data: Dict):
        """Save model prediction"""
        try:
            self.cursor.execute(
                """
                INSERT INTO predictions
                (timestamp, symbol, model_name, prediction, confidence, 
                 probabilities, features)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (prediction_data['timestamp'], prediction_data['symbol'],
                 prediction_data['model_name'], prediction_data['prediction'],
                 prediction_data['confidence'], 
                 psycopg2.extras.Json(prediction_data.get('probabilities', {})),
                 psycopg2.extras.Json(prediction_data.get('features', {})))
            )
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            self.connection.rollback()
    
    def save_performance_metrics(self, metrics: Dict):
        """Save daily performance metrics"""
        try:
            self.cursor.execute(
                """
                INSERT INTO performance_metrics
                (date, total_trades, winning_trades, losing_trades,
                 total_profit, max_drawdown, sharpe_ratio, win_rate, profit_factor)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    winning_trades = EXCLUDED.winning_trades,
                    losing_trades = EXCLUDED.losing_trades,
                    total_profit = EXCLUDED.total_profit,
                    max_drawdown = EXCLUDED.max_drawdown,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    win_rate = EXCLUDED.win_rate,
                    profit_factor = EXCLUDED.profit_factor
                """,
                (metrics['date'], metrics['total_trades'], 
                 metrics['winning_trades'], metrics['losing_trades'],
                 metrics['total_profit'], metrics['max_drawdown'],
                 metrics.get('sharpe_ratio'), metrics.get('win_rate'),
                 metrics.get('profit_factor'))
            )
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
            self.connection.rollback()
    
    def get_trade_history(self, start_date: datetime = None, 
                         end_date: datetime = None) -> pd.DataFrame:
        """Get trade history"""
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND entry_time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND entry_time <= %s"
                params.append(end_date)
            
            query += " ORDER BY entry_time DESC"
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            if rows:
                return pd.DataFrame(rows)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("Database connection closed")