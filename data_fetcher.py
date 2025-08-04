import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import yfinance as yf

class DataFetcher:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def fetch_multiple_symbols(self, symbols: List[str], 
                                   data_type: str = 'ohlcv') -> Dict:
        """Fetch data for multiple symbols concurrently"""
        tasks = []
        
        for symbol in symbols:
            if data_type == 'ohlcv':
                task = self.fetch_ohlcv_data(symbol)
            elif data_type == 'news':
                task = self.fetch_news_data(symbol)
            elif data_type == 'sentiment':
                task = self.fetch_sentiment_data(symbol)
            else:
                continue
                
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return dict(zip(symbols, results))
    
    async def fetch_ohlcv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from various sources"""
        try:
            # Try Yahoo Finance first
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo", interval="1h")
            
            if not data.empty:
                return data
            
            # Fallback to other sources
            # Alpha Vantage
            if 'alpha_vantage' in self.config.get('api_keys', {}):
                data = await self.fetch_alpha_vantage(symbol)
                if data is not None:
                    return data
            
            # Add more data sources as needed
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
            
        return None
    
    async def fetch_alpha_vantage(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        api_key = self.config['api_keys']['alpha_vantage']
        url = f"https://www.alphavantage.co/query"
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': symbol[:3],
            'to_symbol': symbol[3:],
            'interval': '60min',
            'apikey': api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'Time Series FX (60min)' in data:
                        df = pd.DataFrame.from_dict(
                            data['Time Series FX (60min)'], 
                            orient='index'
                        )
                        df.index = pd.to_datetime(df.index)
                        df = df.astype(float)
                        df.columns = ['open', 'high', 'low', 'close']
                        df = df.sort_index()
                        
                        return df
                        
        except Exception as e:
            self.logger.error(f"Alpha Vantage error: {e}")
            
        return None
    
    async def fetch_news_data(self, symbol: str) -> List[Dict]:
        """Fetch news data for symbol"""
        news_data = []
        
        # NewsAPI
        if 'newsapi' in self.config.get('api_keys', {}):
            news = await self.fetch_newsapi(symbol)
            news_data.extend(news)
        
        # Alpha Vantage News
        if 'alpha_vantage' in self.config.get('api_keys', {}):
            news = await self.fetch_alpha_vantage_news(symbol)
            news_data.extend(news)
        
        return news_data
    
    async def fetch_newsapi(self, symbol: str) -> List[Dict]:
        """Fetch news from NewsAPI"""
        api_key = self.config['api_keys']['newsapi']
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'q': f"{symbol} forex currency",
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    articles = []
                    for article in data.get('articles', []):
                        articles.append({
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'publishedAt': article['publishedAt'],
                            'source': article['source']['name']
                        })
                    
                    return articles
                    
        except Exception as e:
            self.logger.error(f"NewsAPI error: {e}")
            
        return []
    
    async def fetch_alpha_vantage_news(self, symbol: str) -> List[Dict]:
        """Fetch news sentiment from Alpha Vantage"""
        api_key = self.config['api_keys']['alpha_vantage']
        url = "https://www.alphavantage.co/query"
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': f"FOREX:{symbol}",
            'apikey': api_key
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    articles = []
                    for item in data.get('feed', [])[:10]:
                        articles.append({
                            'title': item['title'],
                            'description': item['summary'],
                            'url': item['url'],
                            'publishedAt': item['time_published'],
                            'sentiment_score': item.get('overall_sentiment_score', 0),
                            'sentiment_label': item.get('overall_sentiment_label', 'Neutral')
                        })
                    
                    return articles
                    
        except Exception as e:
            self.logger.error(f"Alpha Vantage news error: {e}")
            
        return []
    
    async def fetch_sentiment_data(self, symbol: str) -> Dict:
        """Fetch sentiment data from various sources"""
        sentiment_data = {
            'news_sentiment': 0.5,
            'social_sentiment': 0.5,
            'analyst_sentiment': 0.5
        }
        
        # Aggregate news sentiment
        news = await self.fetch_news_data(symbol)
        if news:
            sentiments = []
            for article in news:
                if 'sentiment_score' in article:
                    sentiments.append(float(article['sentiment_score']))
            
            if sentiments:
                sentiment_data['news_sentiment'] = np.mean(sentiments)
        
        # Add social media sentiment (Twitter, Reddit, etc.)
        # This would require additional API integrations
        
        # Add analyst ratings
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is not None and not recommendations.empty:
                recent = recommendations.tail(5)
                # Convert to numeric sentiment
                sentiment_map = {
                    'Strong Buy': 1.0,
                    'Buy': 0.75,
                    'Hold': 0.5,
                    'Sell': 0.25,
                    'Strong Sell': 0.0
                }
                
                scores = [sentiment_map.get(rec, 0.5) for rec in recent['To Grade']]
                sentiment_data['analyst_sentiment'] = np.mean(scores)
                
        except Exception as e:
            self.logger.error(f"Error fetching analyst sentiment: {e}")
        
        return sentiment_data
    
    async def fetch_economic_calendar(self) -> List[Dict]:
        """Fetch economic calendar events"""
        # This would integrate with economic calendar APIs
        # For now, returning mock data
        
        events = [
            {
                'datetime': datetime.now() + timedelta(hours=2),
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'importance': 'High',
                'forecast': '200K',
                'previous': '187K'
            },
            {
                'datetime': datetime.now() + timedelta(days=1),
                'currency': 'EUR',
                'event': 'ECB Interest Rate Decision',
                'importance': 'High',
                'forecast': '4.00%',
                'previous': '4.00%'
            }
        ]
        
        return events