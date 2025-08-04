import requests
import pandas as pd
from textblob import TextBlob
from transformers import pipeline
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re

class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                         model="ProsusAI/finbert")
        self.news_sources = config.get('news_sources', [])
        self.api_keys = config.get('api_keys', {})
        
    async def get_market_sentiment(self, symbols):
        """Get comprehensive market sentiment"""
        tasks = []
        
        for symbol in symbols:
            tasks.append(self.get_symbol_sentiment(symbol))
        
        sentiments = await asyncio.gather(*tasks)
        
        # Aggregate sentiment
        overall_sentiment = {
            'news_sentiment': np.mean([s['news_sentiment'] for s in sentiments]),
            'social_sentiment': np.mean([s['social_sentiment'] for s in sentiments]),
            'analyst_sentiment': np.mean([s['analyst_sentiment'] for s in sentiments]),
            'combined_sentiment': np.mean([s['combined_sentiment'] for s in sentiments])
        }
        
        return overall_sentiment, sentiments
    
    async def get_symbol_sentiment(self, symbol):
        """Get sentiment for specific symbol"""
        # News sentiment
        news_sentiment = await self.analyze_news_sentiment(symbol)
        
        # Social media sentiment
        social_sentiment = await self.analyze_social_sentiment(symbol)
        
        # Analyst sentiment
        analyst_sentiment = self.analyze_analyst_sentiment(symbol)
        
        # Fear and Greed Index
        fear_greed = await self.get_fear_greed_index()
        
        # Combine sentiments
        combined_sentiment = (
            news_sentiment * 0.4 +
            social_sentiment * 0.3 +
            analyst_sentiment * 0.2 +
            fear_greed * 0.1
        )
        
        return {
            'symbol': symbol,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'analyst_sentiment': analyst_sentiment,
            'fear_greed': fear_greed,
            'combined_sentiment': combined_sentiment,
            'timestamp': datetime.now()
        }
    
    async def analyze_news_sentiment(self, symbol):
        """Analyze news sentiment"""
        news_data = await self.fetch_news(symbol)
        
        if not news_data:
            return 0.5  # Neutral
        
        sentiments = []
        
        for article in news_data:
            # Use FinBERT for financial sentiment
            result = self.sentiment_pipeline(article['title'] + " " + article.get('description', ''))
            
            # Convert to numeric score
            if result[0]['label'] == 'positive':
                score = result[0]['score']
            elif result[0]['label'] == 'negative':
                score = -result[0]['score']
            else:
                score = 0
            
            sentiments.append(score)
        
        # Weight recent news more heavily
        weights = np.exp(-np.arange(len(sentiments)) * 0.1)
        weights /= weights.sum()
        
        weighted_sentiment = np.average(sentiments, weights=weights) if sentiments else 0
        
        # Normalize to 0-1 range
        return (weighted_sentiment + 1) / 2
    
    async def fetch_news(self, symbol):
        """Fetch news from multiple sources"""
        news_data = []
        
        # Alpha Vantage News
        if 'alpha_vantage' in self.api_keys:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_keys['alpha_vantage']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'feed' in data:
                            for item in data['feed'][:10]:
                                news_data.append({
                                    'title': item['title'],
                                    'description': item['summary'],
                                    'sentiment': item.get('overall_sentiment_score', 0)
                                })
        
        return news_data
    
    async def analyze_social_sentiment(self, symbol):
        """Analyze social media sentiment"""
        # This would integrate with Twitter API, Reddit API, etc.
        # For now, returning mock data
        return 0.6
    
    def analyze_analyst_sentiment(self, symbol):
        """Analyze analyst recommendations"""
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is not None and not recommendations.empty:
                recent_recs = recommendations.tail(10)
                
                # Convert recommendations to scores
                rec_scores = {
                    'Strong Buy': 1.0,
                    'Buy': 0.75,
                    'Hold': 0.5,
                    'Sell': 0.25,
                    'Strong Sell': 0.0
                }
                
                scores = [rec_scores.get(rec, 0.5) for rec in recent_recs['To Grade']]
                return np.mean(scores)
            
        except:
            pass
        
        return 0.5  # Neutral if no data
    
    async def get_fear_greed_index(self):
        """Get Fear and Greed Index"""
        # This would fetch from CNN Fear & Greed Index API
        # Returning mock data for now
        return 0.5
    
    def calculate_sentiment_momentum(self, historical_sentiments):
        """Calculate sentiment momentum"""
        if len(historical_sentiments) < 2:
            return 0
        
        recent = historical_sentiments[-5:].mean()
        older = historical_sentiments[-10:-5].mean()
        
        momentum = (recent - older) / (older + 1e-10)
        
        return momentum