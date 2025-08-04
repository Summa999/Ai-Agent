import pandas as pd
import numpy as np
from typing import Dict, List
import yfinance as yf
from datetime import datetime, timedelta
import logging

class FundamentalAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        
    def analyze_currency_fundamentals(self, currency_pair: str) -> Dict:
        """Analyze fundamental factors for currency pair"""
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[3:6]
        
        # Get economic indicators
        base_fundamentals = self.get_currency_fundamentals(base_currency)
        quote_fundamentals = self.get_currency_fundamentals(quote_currency)
        
        # Calculate relative strength
        relative_strength = self.calculate_relative_strength(
            base_fundamentals, quote_fundamentals
        )
        
        # Get central bank data
        cb_analysis = self.analyze_central_banks(base_currency, quote_currency)
        
        # Economic calendar events
        events = self.get_economic_events(base_currency, quote_currency)
        
        return {
            'base_currency': base_fundamentals,
            'quote_currency': quote_fundamentals,
            'relative_strength': relative_strength,
            'central_bank_analysis': cb_analysis,
            'upcoming_events': events,
            'fundamental_bias': self.calculate_fundamental_bias(
                relative_strength, cb_analysis
            )
        }
    
    def get_currency_fundamentals(self, currency: str) -> Dict:
        """Get fundamental data for a currency"""
        # In a real implementation, this would fetch from economic data APIs
        # For now, returning mock data
        
        fundamentals = {
            'gdp_growth': self.get_gdp_data(currency),
            'inflation_rate': self.get_inflation_data(currency),
            'interest_rate': self.get_interest_rate(currency),
            'employment_rate': self.get_employment_data(currency),
            'trade_balance': self.get_trade_balance(currency),
            'debt_to_gdp': self.get_debt_ratio(currency),
            'consumer_confidence': self.get_consumer_confidence(currency),
            'pmi': self.get_pmi_data(currency)
        }
        
        return fundamentals
    
    def get_gdp_data(self, currency: str) -> Dict:
        """Get GDP data for currency's country"""
        # Mock implementation
        gdp_data = {
            'USD': {'current': 2.1, 'previous': 2.0, 'forecast': 2.2},
            'EUR': {'current': 1.5, 'previous': 1.3, 'forecast': 1.6},
            'GBP': {'current': 1.8, 'previous': 1.7, 'forecast': 1.9},
            'JPY': {'current': 0.9, 'previous': 0.8, 'forecast': 1.0},
            'AUD': {'current': 2.5, 'previous': 2.3, 'forecast': 2.6},
        }
        
        return gdp_data.get(currency, {'current': 0, 'previous': 0, 'forecast': 0})
    
    def get_inflation_data(self, currency: str) -> Dict:
        """Get inflation data"""
        inflation_data = {
            'USD': {'current': 3.7, 'previous': 3.5, 'target': 2.0},
            'EUR': {'current': 2.9, 'previous': 2.7, 'target': 2.0},
            'GBP': {'current': 4.0, 'previous': 3.8, 'target': 2.0},
            'JPY': {'current': 2.5, 'previous': 2.3, 'target': 2.0},
            'AUD': {'current': 3.5, 'previous': 3.3, 'target': 2.5},
        }
        
        return inflation_data.get(currency, {'current': 0, 'previous': 0, 'target': 0})
    
    def get_interest_rate(self, currency: str) -> Dict:
        """Get central bank interest rates"""
        rates = {
            'USD': {'current': 5.25, 'previous': 5.00, 'expected': 5.25},
            'EUR': {'current': 4.00, 'previous': 3.75, 'expected': 4.00},
            'GBP': {'current': 5.00, 'previous': 4.75, 'expected': 5.00},
            'JPY': {'current': -0.10, 'previous': -0.10, 'expected': 0.00},
            'AUD': {'current': 4.10, 'previous': 3.85, 'expected': 4.10},
        }
        
        return rates.get(currency, {'current': 0, 'previous': 0, 'expected': 0})
    
    def get_employment_data(self, currency: str) -> Dict:
        """Get employment data"""
        employment = {
            'USD': {'unemployment': 3.8, 'nfp': 236000, 'wage_growth': 4.2},
            'EUR': {'unemployment': 6.5, 'nfp': None, 'wage_growth': 3.5},
            'GBP': {'unemployment': 4.3, 'nfp': None, 'wage_growth': 5.1},
            'JPY': {'unemployment': 2.5, 'nfp': None, 'wage_growth': 2.0},
            'AUD': {'unemployment': 3.7, 'nfp': None, 'wage_growth': 3.8},
        }
        
        return employment.get(currency, {'unemployment': 0, 'nfp': 0, 'wage_growth': 0})
    
    def get_trade_balance(self, currency: str) -> float:
        """Get trade balance data"""
        trade_balance = {
            'USD': -75.3,  # Billion
            'EUR': 15.2,
            'GBP': -20.1,
            'JPY': 10.5,
            'AUD': 8.3,
        }
        
        return trade_balance.get(currency, 0)
    
    def get_debt_ratio(self, currency: str) -> float:
        """Get debt to GDP ratio"""
        debt_ratios = {
            'USD': 123.4,
            'EUR': 84.1,
            'GBP': 101.2,
            'JPY': 264.3,
            'AUD': 47.5,
        }
        
        return debt_ratios.get(currency, 0)
    
    def get_consumer_confidence(self, currency: str) -> float:
        """Get consumer confidence index"""
        confidence = {
            'USD': 102.5,
            'EUR': 95.3,
            'GBP': 98.7,
            'JPY': 88.9,
            'AUD': 105.2,
        }
        
        return confidence.get(currency, 100)
    
    def get_pmi_data(self, currency: str) -> Dict:
        """Get PMI data"""
        pmi = {
            'USD': {'manufacturing': 48.7, 'services': 52.3},
            'EUR': {'manufacturing': 44.2, 'services': 48.7},
            'GBP': {'manufacturing': 46.5, 'services': 49.2},
            'JPY': {'manufacturing': 48.9, 'services': 53.8},
            'AUD': {'manufacturing': 50.2, 'services': 51.5},
        }
        
        return pmi.get(currency, {'manufacturing': 50, 'services': 50})
    
    def calculate_relative_strength(self, base_fundamentals: Dict,
                                  quote_fundamentals: Dict) -> Dict:
        """Calculate relative fundamental strength"""
        strength_score = 0
        factors = {}
        
        # GDP comparison
        gdp_diff = (base_fundamentals['gdp_growth']['current'] - 
                    quote_fundamentals['gdp_growth']['current'])
        strength_score += gdp_diff * 0.2
        factors['gdp'] = gdp_diff
        
        # Interest rate differential
        rate_diff = (base_fundamentals['interest_rate']['current'] - 
                    quote_fundamentals['interest_rate']['current'])
        strength_score += rate_diff * 0.3
        factors['interest_rate'] = rate_diff
        
        # Inflation differential
        inflation_diff = (quote_fundamentals['inflation_rate']['current'] - 
                         base_fundamentals['inflation_rate']['current'])
        strength_score += inflation_diff * 0.15
        factors['inflation'] = inflation_diff
        
        # Employment
        employment_diff = (quote_fundamentals['employment_rate']['unemployment'] - 
                          base_fundamentals['employment_rate']['unemployment'])
        strength_score += employment_diff * 0.15
        factors['employment'] = employment_diff
        
        # Trade balance
        trade_diff = (base_fundamentals['trade_balance'] - 
                     quote_fundamentals['trade_balance']) / 100
        strength_score += trade_diff * 0.1
        factors['trade'] = trade_diff
        
        # Debt levels
        debt_diff = (quote_fundamentals['debt_to_gdp'] - 
                    base_fundamentals['debt_to_gdp']) / 100
        strength_score += debt_diff * 0.1
        factors['debt'] = debt_diff
        
        return {
            'total_score': strength_score,
            'factors': factors,
            'bias': 'bullish' if strength_score > 0.5 else 'bearish' if strength_score < -0.5 else 'neutral'
        }
    
    def analyze_central_banks(self, base_currency: str, quote_currency: str) -> Dict:
        """Analyze central bank policies"""
        base_cb = self.get_central_bank_stance(base_currency)
        quote_cb = self.get_central_bank_stance(quote_currency)
        
        return {
            'base_currency_cb': base_cb,
            'quote_currency_cb': quote_cb,
            'policy_divergence': self.calculate_policy_divergence(base_cb, quote_cb)
        }
    
    def get_central_bank_stance(self, currency: str) -> Dict:
        """Get central bank stance"""
        cb_stance = {
            'USD': {
                'stance': 'hawkish',
                'next_meeting': '2024-03-20',
                'expected_action': 'hold',
                'tone_score': 0.7
            },
            'EUR': {
                'stance': 'neutral',
                'next_meeting': '2024-03-07',
                'expected_action': 'hold',
                'tone_score': 0.0
            },
            'GBP': {
                'stance': 'hawkish',
                'next_meeting': '2024-03-21',
                'expected_action': 'hold',
                'tone_score': 0.6
            },
            'JPY': {
                'stance': 'dovish',
                'next_meeting': '2024-03-19',
                'expected_action': 'hold',
                'tone_score': -0.5
            },
            'AUD': {
                'stance': 'neutral',
                'next_meeting': '2024-03-05',
                'expected_action': 'hold',
                'tone_score': 0.1
            }
        }
        
        return cb_stance.get(currency, {
            'stance': 'neutral',
            'next_meeting': 'unknown',
            'expected_action': 'hold',
            'tone_score': 0.0
        })
    
    def calculate_policy_divergence(self, base_cb: Dict, quote_cb: Dict) -> float:
        """Calculate monetary policy divergence"""
        return base_cb['tone_score'] - quote_cb['tone_score']
    
    def get_economic_events(self, base_currency: str, quote_currency: str) -> List[Dict]:
        """Get upcoming economic events"""
        # In real implementation, this would fetch from economic calendar API
        events = [
            {
                'date': '2024-01-15',
                'time': '13:30',
                'currency': base_currency,
                'event': 'CPI',
                'importance': 'high',
                'forecast': 3.2,
                'previous': 3.1
            },
            {
                'date': '2024-01-16',
                'time': '10:00',
                'currency': quote_currency,
                'event': 'GDP',
                'importance': 'high',
                'forecast': 1.8,
                'previous': 1.5
            }
        ]
        
        return events
    
    def calculate_fundamental_bias(self, relative_strength: Dict,
                                 cb_analysis: Dict) -> Dict:
        """Calculate overall fundamental bias"""
        # Combine relative strength and CB divergence
        fundamental_score = (relative_strength['total_score'] * 0.7 +
                           cb_analysis['policy_divergence'] * 0.3)
        
        if fundamental_score > 0.5:
            bias = 'strong_bullish'
            confidence = min(fundamental_score, 1.0)
        elif fundamental_score > 0.2:
            bias = 'bullish'
            confidence = fundamental_score / 0.5
        elif fundamental_score < -0.5:
            bias = 'strong_bearish'
            confidence = min(abs(fundamental_score), 1.0)
        elif fundamental_score < -0.2:
            bias = 'bearish'
            confidence = abs(fundamental_score) / 0.5
        else:
            bias = 'neutral'
            confidence = 1 - abs(fundamental_score) / 0.2
        
        return {
            'bias': bias,
            'score': fundamental_score,
            'confidence': confidence
        }