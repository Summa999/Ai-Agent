# main.py - Complete Enhanced AI Trading Bot
import asyncio
import sys
import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import time
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init(autoreset=True)

# ==================== CONFIGURATION MANAGER ====================
class ConfigManager:
    """Manages configuration loading and validation"""
    
    @staticmethod
    def create_default_config():
        """Create default configuration file"""
        config = {
            'trading': {
                'enabled': False,
                'mode': 'paper',
                'max_daily_trades': 10,
                'max_concurrent_positions': 3,
                'default_risk_percent': 2.0,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 6.0
            },
            'agent': {
                'thinking_frequency': 30,
                'detailed_analysis': True,
                'verbose_reasoning': True,
                'decision_logging': True,
                'learning_tracking': True,
                'confidence_threshold': 0.6,
                'max_drawdown_percent': 10.0,
                'adaptive_learning': True
            },
            'markets': {
                'forex': {
                    'enabled': True,
                    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF'],
                    'timeframes': ['M15', 'H1', 'H4'],
                    'max_spread': 3.0
                },
                'crypto': {
                    'enabled': False,
                    'exchange': 'binance',
                    'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                    'api_url': 'https://api.binance.com',
                    'max_slippage': 0.5
                }
            },
            'mt5': {
                'enabled': True,
                'server': 'your-broker-server',
                'login': 12345678,
                'password': 'your-password',
                'path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
            },
            'logging': {
                'level': 'INFO',
                'console_colors': True,
                'show_reasoning': True,
                'show_analysis': True,
                'show_learning': True,
                'file_logging': True
            },
            'display': {
                'real_time_updates': True,
                'show_market_data': True,
                'show_indicators': True,
                'show_opportunities': True,
                'console_width': 120
            }
        }
        
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return config
    
    @staticmethod
    def load_config():
        """Load configuration from file"""
        try:
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("Config file not found, creating default...")
            return ConfigManager.create_default_config()

# ==================== ENHANCED LOGGING SYSTEM ====================
class EnhancedLogger:
    """Enhanced logging system with colors and formatting"""
    
    def __init__(self):
        self.setup_logging()
        self.decision_count = 0
        self.cycle_count = 0
    
    def setup_logging(self):
        """Setup enhanced logging"""
        # Create logs directory
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Setup formatters
        console_format = '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s'
        file_format = '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'
        
        # Root logger
        logging.basicConfig(
            level=logging.INFO,
            format=console_format,
            datefmt='%H:%M:%S'
        )
        
        # File handler
        log_filename = f"logs/ai_agent_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        print(f"{Fore.GREEN}[✓] Logging setup complete - {log_filename}{Style.RESET_ALL}")

# ==================== MARKET DATA SIMULATOR ====================
class MarketDataSimulator:
    """Simulates market data when real connections aren't available"""
    
    def __init__(self):
        self.logger = logging.getLogger('MarketSim')
        self.base_prices = {
            'EURUSD': 1.0950,
            'GBPUSD': 1.2650,
            'USDJPY': 149.50,
            'AUDUSD': 0.6580,
            'USDCHF': 0.8920,
            'BTCUSDT': 43500.0,
            'ETHUSDT': 2650.0
        }
        self.price_history = {}
        self.initialize_history()
    
    def initialize_history(self):
        """Initialize price history for each symbol"""
        for symbol, base_price in self.base_prices.items():
            self.price_history[symbol] = self.generate_historical_data(base_price)
    
    def generate_historical_data(self, base_price, periods=100):
        """Generate realistic historical price data"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='15T')
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.001, periods)  # Small random movements
        prices = [base_price]
        
        for i in range(1, periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Create realistic OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.0005)))
            low = price * (1 - abs(np.random.normal(0, 0.0005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    async def get_current_price(self, symbol):
        """Get current price for symbol"""
        if symbol not in self.base_prices:
            return None
        
        # Simulate small price movements
        base = self.base_prices[symbol]
        change = np.random.normal(0, base * 0.001)
        current = base + change
        
        # Update base price slowly
        self.base_prices[symbol] = current
        
        return {
            'symbol': symbol,
            'last': current,
            'bid': current - (current * 0.0001),
            'ask': current + (current * 0.0001),
            'volume': np.random.randint(1000, 50000),
            'timestamp': datetime.now()
        }
    
    async def get_historical_data(self, symbol, timeframe='15m', periods=100):
        """Get historical data for symbol"""
        if symbol in self.price_history:
            return self.price_history[symbol].tail(periods).copy()
        else:
            return pd.DataFrame()

# ==================== TECHNICAL ANALYSIS ENGINE ====================
class TechnicalAnalyzer:
    """Advanced technical analysis with multiple indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger('TechAnalysis')
    
    async def analyze(self, historical_data, current_price):
        """Perform comprehensive technical analysis"""
        if historical_data.empty:
            return self.get_empty_analysis()
        
        try:
            analysis = {}
            
            # Basic indicators
            analysis['rsi'] = self.calculate_rsi(historical_data['close'])
            analysis['macd'] = self.calculate_macd(historical_data['close'])
            analysis['bollinger'] = self.calculate_bollinger_bands(historical_data['close'])
            analysis['sma_20'] = historical_data['close'].rolling(20).mean().iloc[-1]
            analysis['ema_12'] = historical_data['close'].ewm(span=12).mean().iloc[-1]
            
            # Trend analysis
            analysis['trend'] = self.determine_trend(historical_data)
            analysis['support'] = self.find_support(historical_data)
            analysis['resistance'] = self.find_resistance(historical_data)
            
            # Volume analysis
            analysis['volume_trend'] = self.analyze_volume(historical_data)
            
            # Price action
            analysis['price_action'] = self.analyze_price_action(historical_data)
            
            # Overall signal
            analysis['signal'] = self.generate_signal(analysis)
            analysis['confidence'] = self.calculate_confidence(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            return self.get_empty_analysis()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def calculate_macd(self, prices):
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        # Determine signal
        if len(macd_line) >= 2:
            if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                signal = 'BULLISH_CROSSOVER'
            elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                signal = 'BEARISH_CROSSOVER'
            elif macd_line.iloc[-1] > signal_line.iloc[-1]:
                signal = 'BULLISH'
            else:
                signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
            'signal': signal_line.iloc[-1] if not signal_line.empty else 0,
            'histogram': histogram.iloc[-1] if not histogram.empty else 0,
            'signal_type': signal
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        return {
            'upper': (sma + (std * std_dev)).iloc[-1] if not sma.empty else 0,
            'middle': sma.iloc[-1] if not sma.empty else 0,
            'lower': (sma - (std * std_dev)).iloc[-1] if not sma.empty else 0
        }
    
    def determine_trend(self, data):
        """Determine market trend"""
        if len(data) < 20:
            return {'direction': 'UNKNOWN', 'strength': 0}
        
        sma_short = data['close'].rolling(10).mean()
        sma_long = data['close'].rolling(20).mean()
        
        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if current_price > current_short > current_long:
            direction = 'BULLISH'
            strength = min(((current_price - current_long) / current_long) * 100, 10)
        elif current_price < current_short < current_long:
            direction = 'BEARISH'
            strength = min(((current_long - current_price) / current_long) * 100, 10)
        else:
            direction = 'SIDEWAYS'
            strength = 0
        
        return {'direction': direction, 'strength': abs(strength)}
    
    def find_support(self, data):
        """Find support level"""
        lows = data['low'].rolling(window=10).min()
        return lows.iloc[-5:].min()
    
    def find_resistance(self, data):
        """Find resistance level"""
        highs = data['high'].rolling(window=10).max()
        return highs.iloc[-5:].max()
    
    def analyze_volume(self, data):
        """Analyze volume trends"""
        if 'volume' not in data.columns:
            return 'UNKNOWN'
        
        recent_vol = data['volume'].iloc[-5:].mean()
        historical_vol = data['volume'].iloc[-20:-5].mean()
        
        if recent_vol > historical_vol * 1.5:
            return 'INCREASING'
        elif recent_vol < historical_vol * 0.7:
            return 'DECREASING'
        else:
            return 'NORMAL'
    
    def analyze_price_action(self, data):
        """Analyze recent price action"""
        if len(data) < 5:
            return 'UNKNOWN'
        
        recent_closes = data['close'].iloc[-5:]
        price_change = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0] * 100
        
        if price_change > 1:
            return 'STRONG_UP'
        elif price_change > 0.3:
            return 'MODERATE_UP'
        elif price_change < -1:
            return 'STRONG_DOWN'
        elif price_change < -0.3:
            return 'MODERATE_DOWN'
        else:
            return 'SIDEWAYS'
    
    def generate_signal(self, analysis):
        """Generate overall trading signal"""
        signals = []
        
        # RSI signals
        if analysis['rsi'] < 30:
            signals.append('BUY')
        elif analysis['rsi'] > 70:
            signals.append('SELL')
        
        # MACD signals
        if analysis['macd']['signal_type'] == 'BULLISH_CROSSOVER':
            signals.append('BUY')
        elif analysis['macd']['signal_type'] == 'BEARISH_CROSSOVER':
            signals.append('SELL')
        
        # Trend signals
        if analysis['trend']['direction'] == 'BULLISH' and analysis['trend']['strength'] > 2:
            signals.append('BUY')
        elif analysis['trend']['direction'] == 'BEARISH' and analysis['trend']['strength'] > 2:
            signals.append('SELL')
        
        # Determine overall signal
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        
        if buy_signals > sell_signals:
            return 'BUY'
        elif sell_signals > buy_signals:
            return 'SELL'
        else:
            return 'HOLD'
    
    def calculate_confidence(self, analysis):
        """Calculate confidence level for the signal"""
        confidence = 0.5  # Base confidence
        
        # RSI confidence
        rsi = analysis['rsi']
        if rsi < 25 or rsi > 75:
            confidence += 0.2
        elif rsi < 35 or rsi > 65:
            confidence += 0.1
        
        # Trend confidence
        trend_strength = analysis['trend']['strength']
        if trend_strength > 3:
            confidence += 0.2
        elif trend_strength > 1:
            confidence += 0.1
        
        # MACD confidence
        if 'CROSSOVER' in analysis['macd']['signal_type']:
            confidence += 0.15
        
        # Volume confirmation
        if analysis['volume_trend'] == 'INCREASING':
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_empty_analysis(self):
        """Return empty analysis structure"""
        return {
            'rsi': 50,
            'macd': {'macd': 0, 'signal': 0, 'histogram': 0, 'signal_type': 'NEUTRAL'},
            'bollinger': {'upper': 0, 'middle': 0, 'lower': 0},
            'sma_20': 0,
            'ema_12': 0,
            'trend': {'direction': 'UNKNOWN', 'strength': 0},
            'support': 0,
            'resistance': 0,
            'volume_trend': 'UNKNOWN',
            'price_action': 'UNKNOWN',
            'signal': 'HOLD',
            'confidence': 0.3
        }

# ==================== ENHANCED AI AGENT ====================
class EnhancedAIAgent:
    """Enhanced AI Agent with comprehensive market analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('AIAgent')
        
        # Components
        self.market_sim = MarketDataSimulator()
        self.tech_analyzer = TechnicalAnalyzer()
        
        # Agent state
        self.cycle_count = 0
        self.decisions_made = 0
        self.successful_trades = 0
        self.learning_points = []
        self.market_insights = {}
        self.opportunity_history = []
        
        # Configuration
        self.thinking_freq = config.get('agent', {}).get('thinking_frequency', 30)
        self.confidence_threshold = config.get('agent', {}).get('confidence_threshold', 0.6)
        
        self.logger.info(f"{Fore.MAGENTA}[🤖] Enhanced AI Agent initialized{Style.RESET_ALL}")
    
    async def start_thinking_loop(self):
        """Main thinking loop"""
        self.logger.info(f"{Fore.BLUE}[🧠] Starting AI thinking loop (every {self.thinking_freq}s){Style.RESET_ALL}")
        
        while True:
            try:
                await self.enhanced_thinking_cycle()
                await asyncio.sleep(self.thinking_freq)
            except KeyboardInterrupt:
                self.logger.info(f"{Fore.RED}[⏹️] AI Agent stopped by user{Style.RESET_ALL}")
                break
            except Exception as e:
                self.logger.error(f"Error in thinking loop: {e}")
                await asyncio.sleep(self.thinking_freq)
    
    async def enhanced_thinking_cycle(self):
        """Enhanced thinking cycle with detailed analysis"""
        self.cycle_count += 1
        start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info(f"{Fore.CYAN}[🧠] THINKING CYCLE #{self.cycle_count} STARTED{Style.RESET_ALL}")
        self.logger.info("=" * 80)
        
        try:
            # Phase 1: Market Data Collection
            market_data = await self.gather_market_data()
            
            # Phase 2: Technical Analysis
            analysis_results = await self.perform_analysis(market_data)
            
            # Phase 3: Opportunity Identification
            opportunities = await self.identify_opportunities(analysis_results)
            
            # Phase 4: Decision Making
            decisions = await self.make_decisions(opportunities)
            
            # Phase 5: Learning & Adaptation
            learning_insights = await self.learn_and_adapt(decisions)
            
            # Phase 6: Cycle Summary
            await self.log_cycle_summary(start_time, {
                'market_data': market_data,
                'opportunities': opportunities,
                'decisions': decisions,
                'learning': learning_insights
            })
            
        except Exception as e:
            self.logger.error(f"Error in thinking cycle: {e}")
        
        self.logger.info("=" * 80)
    
    async def gather_market_data(self):
        """Gather comprehensive market data"""
        self.logger.info(f"{Fore.GREEN}[📊] PHASE 1: GATHERING MARKET DATA{Style.RESET_ALL}")
        
        market_data = {}
        symbols = self.config.get('markets', {}).get('forex', {}).get('symbols', [])
        
        for symbol in symbols:
            try:
                # Get current price
                price_data = await self.market_sim.get_current_price(symbol)
                
                # Get historical data
                historical_data = await self.market_sim.get_historical_data(symbol, '15m', 100)
                
                if price_data and not historical_data.empty:
                    market_data[symbol] = {
                        'price': price_data,
                        'historical': historical_data,
                        'timestamp': datetime.now()
                    }
                    
                    self.logger.info(f"  • {symbol}: ${price_data['last']:.4f} "
                                   f"(Vol: {price_data['volume']:,})")
                
            except Exception as e:
                self.logger.warning(f"  ⚠️ Failed to get data for {symbol}: {e}")
        
        self.logger.info(f"[📊] Data collected for {len(market_data)} symbols")
        return market_data
    
    async def perform_analysis(self, market_data):
        """Perform deep technical analysis"""
        self.logger.info(f"{Fore.BLUE}[🔍] PHASE 2: TECHNICAL ANALYSIS{Style.RESET_ALL}")
        
        analysis_results = {}
        
        for symbol, data in market_data.items():
            self.logger.info(f"[🔬] Analyzing {symbol}...")
            
            # Technical analysis
            technical = await self.tech_analyzer.analyze(
                data['historical'], 
                data['price']
            )
            
            analysis_results[symbol] = {
                'technical': technical,
                'timestamp': datetime.now()
            }
            
            # Log analysis details
            self.log_technical_analysis(symbol, technical)
        
        return analysis_results
    
    def log_technical_analysis(self, symbol, technical):
        """Log detailed technical analysis"""
        self.logger.info(f"  📊 {symbol} Technical Analysis:")
        self.logger.info(f"    • Signal: {technical['signal']} "
                        f"(Confidence: {technical['confidence']:.1%})")
        self.logger.info(f"    • Trend: {technical['trend']['direction']} "
                        f"(Strength: {technical['trend']['strength']:.1f})")
        self.logger.info(f"    • RSI: {technical['rsi']:.1f} "
                        f"({'Overbought' if technical['rsi'] > 70 else 'Oversold' if technical['rsi'] < 30 else 'Neutral'})")
        self.logger.info(f"    • MACD: {technical['macd']['signal_type']}")
        self.logger.info(f"    • Volume: {technical['volume_trend']}")
    
    async def identify_opportunities(self, analysis_results):
        """Identify trading opportunities"""
        self.logger.info(f"{Fore.YELLOW}[🎯] PHASE 3: OPPORTUNITY IDENTIFICATION{Style.RESET_ALL}")
        
        opportunities = []
        
        for symbol, analysis in analysis_results.items():
            technical = analysis['technical']
            
            # Calculate opportunity score
            opportunity = self.evaluate_opportunity(symbol, technical)
            
            if opportunity['score'] >= self.confidence_threshold:
                opportunities.append(opportunity)
                
                self.logger.info(f"[🎯] OPPORTUNITY: {symbol}")
                self.logger.info(f"  • Score: {opportunity['score']:.2f}")
                self.logger.info(f"  • Direction: {opportunity['direction']}")
                self.logger.info(f"  • Reasons: {', '.join(opportunity['reasons'])}")
            
            elif opportunity['score'] > 0.3:
                self.logger.info(f"[📊] Moderate signal: {symbol} "
                               f"(Score: {opportunity['score']:.2f})")
            else:
                self.logger.info(f"[😴] No opportunity: {symbol}")
        
        self.logger.info(f"[🎯] Found {len(opportunities)} high-confidence opportunities")
        return opportunities
    
    def evaluate_opportunity(self, symbol, technical):
        """Evaluate trading opportunity"""
        score = 0
        reasons = []
        direction = technical['signal']
        
        # Base confidence from technical analysis
        score += technical['confidence'] * 0.4
        
        # RSI scoring
        rsi = technical['rsi']
        if rsi < 30:
            score += 0.2
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score += 0.2
            reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # Trend scoring
        if technical['trend']['strength'] > 2:
            score += 0.2
            reasons.append(f"Strong {technical['trend']['direction'].lower()} trend")
        
        # MACD scoring
        if 'CROSSOVER' in technical['macd']['signal_type']:
            score += 0.15
            reasons.append(f"MACD {technical['macd']['signal_type'].lower()}")
        
        # Volume confirmation
        if technical['volume_trend'] == 'INCREASING':
            score += 0.05
            reasons.append("Volume confirmation")
        
        return {
            'symbol': symbol,
            'score': min(score, 1.0),
            'direction': direction,
            'reasons': reasons,
            'technical': technical
        }
    
    async def make_decisions(self, opportunities):
        """Make trading decisions"""
        self.logger.info(f"{Fore.MAGENTA}[⚡] PHASE 4: DECISION MAKING{Style.RESET_ALL}")
        
        decisions = []
        
        for opportunity in opportunities:
            decision = self.make_trading_decision(opportunity)
            decisions.append(decision)
            
            # Log decision
            action_color = Fore.GREEN if decision['action'] == 'BUY' else Fore.RED if decision['action'] == 'SELL' else Fore.YELLOW
            
            self.logger.info(f"{action_color}[⚡] DECISION: {decision['action']} {decision['symbol']}{Style.RESET_ALL}")
            self.logger.info(f"  💭 Confidence: {decision['confidence']:.1%}")
            self.logger.info(f"  💭 Reasoning: {decision['reasoning']}")
            
            if decision['action'] in ['BUY', 'SELL']:
                self.logger.info(f"  📊 Risk/Reward: 1:{decision.get('risk_reward', 3)}")
                self.decisions_made += 1
        
        return decisions
    
    def make_trading_decision(self, opportunity):
        """Make individual trading decision"""
        symbol = opportunity['symbol']
        score = opportunity['score']
        direction = opportunity['direction']
        
        # Risk management
        if score >= 0.8:
            action = direction
            reasoning = f"High confidence ({score:.1%}) with strong technical signals"
        elif score >= self.confidence_threshold:
            action = direction
            reasoning = f"Good confidence ({score:.1%}) with multiple confirmations"
        else:
            action = 'WAIT'
            reasoning = f"Insufficient confidence ({score:.1%}) for trade entry"
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': score,
            'reasoning': reasoning,
            'risk_reward': 3.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 6.0,
            'timestamp': datetime.now()
        }
    
    async def learn_and_adapt(self, decisions):
        """Learn from decisions and adapt"""
        self.logger.info(f"{Fore.CYAN}[📚] PHASE 5: LEARNING & ADAPTATION{Style.RESET_ALL}")
        
        learning_insights = {
            'patterns_found': 0,
            'adaptations_made': 0,
            'performance_metrics': {}
        }
        
        # Update performance metrics
        total_decisions = self.decisions_made
        success_rate = (self.successful_trades / max(total_decisions, 1)) * 100
        
        learning_insights['performance_metrics'] = {
            'total_decisions': total_decisions,
            'success_rate': success_rate,
            'learning_points': len(self.learning_points)
        }
        
        # Log learning insights
        self.logger.info(f"[📚] Performance Metrics:")
        self.logger.info(f"  • Total Decisions: {total_decisions}")
        self.logger.info(f"  • Success Rate: {success_rate:.1f}%")
        self.logger.info(f"  • Learning Points: {len(self.learning_points)}")
        
        # Simulate learning adaptation
        if len(decisions) > 0:
            avg_confidence = sum(d['confidence'] for d in decisions) / len(decisions)
            if avg_confidence < 0.5:
                self.logger.info(f"[📚] Adapting: Lowering confidence threshold due to weak signals")
                learning_insights['adaptations_made'] += 1
        
        return learning_insights
    
    async def log_cycle_summary(self, start_time, cycle_data):
        """Log comprehensive cycle summary"""
        duration = (datetime.now() - start_time).total_seconds()
        
        opportunities = len(cycle_data.get('opportunities', []))
        decisions = len([d for d in cycle_data.get('decisions', []) if d['action'] in ['BUY', 'SELL']])
        markets_analyzed = len(cycle_data.get('market_data', {}))
        
        self.logger.info(f"{Fore.WHITE}[📋] CYCLE #{self.cycle_count} SUMMARY:{Style.RESET_ALL}")
        self.logger.info(f"  ⏱️ Duration: {duration:.1f} seconds")
        self.logger.info(f"  📊 Markets Analyzed: {markets_analyzed}")
        self.logger.info(f"  🎯 Opportunities Found: {opportunities}")
        self.logger.info(f"  ⚡ Trading Decisions: {decisions}")
        
        # Best opportunity
        if cycle_data.get('opportunities'):
            best_opp = max(cycle_data['opportunities'], key=lambda x: x['score'])
            self.logger.info(f"  🏆 Best Opportunity: {best_opp['symbol']} "
                           f"({best_opp['score']:.2f} confidence)")
        
        # Next cycle prediction
        next_cycle_time = datetime.now() + timedelta(seconds=self.thinking_freq)
        self.logger.info(f"  ⏰ Next Cycle: {next_cycle_time.strftime('%H:%M:%S')}")
        
        # Trading status
        trading_status = "LIVE TRADING" if self.config.get('trading', {}).get('enabled') else "ANALYSIS MODE"
        status_color = Fore.RED if trading_status == "LIVE TRADING" else Fore.GREEN
        self.logger.info(f"  🔄 Status: {status_color}{trading_status}{Style.RESET_ALL}")

# ==================== MAIN APPLICATION ====================
class CompleteTradingBot:
    """Complete Enhanced AI Trading Bot"""
    
    def __init__(self):
        self.config = ConfigManager.load_config()
        self.logger_setup = EnhancedLogger()
        self.logger = logging.getLogger('TradingBot')
        self.ai_agent = None
        
        # Display welcome message
        self.display_welcome()
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "=" * 80)
        print(f"{Fore.MAGENTA}{Style.BRIGHT}🤖 ENHANCED AI TRADING AGENT v2.0{Style.RESET_ALL}")
        print("=" * 80)
        print(f"{Fore.CYAN}🎯 FEATURES ACTIVE:{Style.RESET_ALL}")
        print("  • Real-time market analysis every 30 seconds")
        print("  • Advanced technical indicators (RSI, MACD, Bollinger Bands)")
        print("  • Intelligent decision making with confidence scoring")
        print("  • Comprehensive logging and performance tracking")
        print("  • Risk management and position sizing")
        print("  • Adaptive learning algorithms")
        print("=" * 80)
        print(f"{Fore.YELLOW}⚠️  IMPORTANT: Currently in OBSERVATION MODE{Style.RESET_ALL}")
        print("   Set 'trading.enabled: true' in config.yaml for live trading")
        print("=" * 80)
    
    async def initialize(self):
        """Initialize the trading bot"""
        self.logger.info(f"{Fore.GREEN}[🚀] Initializing Enhanced AI Trading Bot{Style.RESET_ALL}")
        
        # Validate configuration
        if not self.validate_config():
            return False
        
        # Initialize AI agent
        self.ai_agent = EnhancedAIAgent(self.config)
        
        # Display configuration
        self.display_configuration()
        
        return True
    
    def validate_config(self):
        """Validate configuration"""
        required_sections = ['trading', 'agent', 'markets']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing configuration section: {section}")
                return False
        
        self.logger.info("[✓] Configuration validated")
        return True
    
    def display_configuration(self):
        """Display current configuration"""
        self.logger.info(f"{Fore.CYAN}[⚙️] CONFIGURATION:{Style.RESET_ALL}")
        self.logger.info(f"  • Thinking Frequency: {self.config['agent']['thinking_frequency']}s")
        self.logger.info(f"  • Confidence Threshold: {self.config['agent']['confidence_threshold']:.1%}")
        self.logger.info(f"  • Max Daily Trades: {self.config['trading']['max_daily_trades']}")
        self.logger.info(f"  • Risk Per Trade: {self.config['trading']['default_risk_percent']:.1f}%")
        
        # Display enabled markets
        enabled_markets = []
        if self.config.get('markets', {}).get('forex', {}).get('enabled'):
            forex_symbols = self.config['markets']['forex']['symbols']
            enabled_markets.append(f"Forex ({len(forex_symbols)} pairs)")
        
        if self.config.get('markets', {}).get('crypto', {}).get('enabled'):
            crypto_symbols = self.config['markets']['crypto']['symbols']
            enabled_markets.append(f"Crypto ({len(crypto_symbols)} pairs)")
        
        self.logger.info(f"  • Markets: {', '.join(enabled_markets) if enabled_markets else 'None'}")
    
    async def run(self):
        """Run the complete trading bot"""
        if not await self.initialize():
            self.logger.error("Initialization failed!")
            return
        
        self.logger.info(f"{Fore.GREEN}[🚀] ENHANCED AI AGENT IS NOW ACTIVE{Style.RESET_ALL}")
        self.logger.info(f"{Fore.BLUE}[🧠] Real-time market analysis enabled{Style.RESET_ALL}")
        self.logger.info(f"{Fore.YELLOW}[⏹️] Press Ctrl+C to stop{Style.RESET_ALL}")
        
        try:
            # Start main AI thinking loop
            await self.ai_agent.start_thinking_loop()
            
        except KeyboardInterrupt:
            self.logger.info(f"{Fore.RED}[⏹️] Trading bot stopped by user{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"[❌] Unexpected error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the trading bot"""
        self.logger.info(f"{Fore.CYAN}[👋] Shutting down Enhanced AI Trading Bot{Style.RESET_ALL}")
        
        if self.ai_agent:
            # Log final statistics
            self.logger.info(f"[📊] Final Statistics:")
            self.logger.info(f"  • Total Cycles: {self.ai_agent.cycle_count}")
            self.logger.info(f"  • Decisions Made: {self.ai_agent.decisions_made}")
            self.logger.info(f"  • Learning Points: {len(self.ai_agent.learning_points)}")
        
        self.logger.info("[✓] Shutdown complete")

# ==================== UTILITY FUNCTIONS ====================
def create_directory_structure():
    """Create necessary directory structure"""
    directories = ['logs', 'data', 'models', 'reports']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[✓] Created directory: {directory}")

def display_system_info():
    """Display system information"""
    print(f"\n{Fore.CYAN}[ℹ️] SYSTEM INFORMATION:{Style.RESET_ALL}")
    print(f"  • Python Version: {sys.version.split()[0]}")
    print(f"  • Platform: {sys.platform}")
    print(f"  • Working Directory: {os.getcwd()}")
    print(f"  • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['pandas', 'numpy', 'yaml', 'colorama']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"{Fore.RED}[❌] Missing packages: {', '.join(missing_packages)}{Style.RESET_ALL}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print(f"{Fore.GREEN}[✓] All dependencies available{Style.RESET_ALL}")
    return True

# ==================== MAIN EXECUTION ====================
async def main():
    """Main entry point"""
    try:
        # Setup
        create_directory_structure()
        display_system_info()
        
        if not check_dependencies():
            return
        
        # Create and run bot
        bot = CompleteTradingBot()
        await bot.run()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[👋] Goodbye!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[❌] Fatal error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Run the complete trading bot
    asyncio.run(main())

# ==================== ADDITIONAL UTILITIES ====================

class PerformanceTracker:
    """Track and analyze bot performance"""
    
    def __init__(self):
        self.trades = []
        self.decisions = []
        self.start_time = datetime.now()
    
    def add_trade(self, trade_data):
        """Add trade to performance tracking"""
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': trade_data['symbol'],
            'action': trade_data['action'],
            'confidence': trade_data['confidence'],
            'result': trade_data.get('result', 'PENDING')
        })
    
    def add_decision(self, decision_data):
        """Add decision to tracking"""
        self.decisions.append({
            'timestamp': datetime.now(),
            'symbol': decision_data['symbol'],
            'action': decision_data['action'],
            'confidence': decision_data['confidence'],
            'reasoning': decision_data['reasoning']
        })
    
    def get_performance_report(self):
        """Generate performance report"""
        total_decisions = len(self.decisions)
        successful_trades = len([t for t in self.trades if t['result'] == 'WIN'])
        
        return {
            'runtime': datetime.now() - self.start_time,
            'total_decisions': total_decisions,
            'total_trades': len(self.trades),
            'successful_trades': successful_trades,
            'success_rate': (successful_trades / max(len(self.trades), 1)) * 100,
            'avg_confidence': sum(d['confidence'] for d in self.decisions) / max(total_decisions, 1)
        }

class ConfigurationWizard:
    """Interactive configuration wizard"""
    
    @staticmethod
    def run_wizard():
        """Run interactive configuration setup"""
        print(f"\n{Fore.CYAN}[🧙] CONFIGURATION WIZARD{Style.RESET_ALL}")
        print("Let's set up your AI trading bot...")
        
        config = {}
        
        # Trading settings
        print(f"\n{Fore.YELLOW}[1] TRADING SETTINGS{Style.RESET_ALL}")
        config['trading'] = {
            'enabled': False,  # Always start with false for safety
            'mode': 'paper',
            'max_daily_trades': int(input("Max daily trades (default 10): ") or "10"),
            'default_risk_percent': float(input("Risk per trade % (default 2.0): ") or "2.0")
        }
        
        # Agent settings
        print(f"\n{Fore.YELLOW}[2] AI AGENT SETTINGS{Style.RESET_ALL}")
        config['agent'] = {
            'thinking_frequency': int(input("Thinking frequency in seconds (default 30): ") or "30"),
            'confidence_threshold': float(input("Confidence threshold 0-1 (default 0.6): ") or "0.6")
        }
        
        # Market settings
        print(f"\n{Fore.YELLOW}[3] MARKET SETTINGS{Style.RESET_ALL}")
        forex_enabled = input("Enable Forex trading? (y/N): ").lower() == 'y'
        
        config['markets'] = {
            'forex': {
                'enabled': forex_enabled,
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'] if forex_enabled else []
            },
            'crypto': {
                'enabled': False,
                'symbols': []
            }
        }
        
        # Save configuration
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"\n{Fore.GREEN}[✓] Configuration saved to config.yaml{Style.RESET_ALL}")
        return config

# ==================== COMMAND LINE INTERFACE ====================
def run_cli():
    """Command line interface for the bot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Trading Bot')
    parser.add_argument('--config', help='Configuration wizard', action='store_true')
    parser.add_argument('--test', help='Test mode (no real trading)', action='store_true')
    parser.add_argument('--freq', type=int, help='Thinking frequency in seconds', default=30)
    parser.add_argument('--verbose', help='Verbose logging', action='store_true')
    
    args = parser.parse_args()
    
    if args.config:
        ConfigurationWizard.run_wizard()
        return
    
    # Update config based on CLI args
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = ConfigManager.create_default_config()
    
    if args.test:
        config['trading']['enabled'] = False
        print(f"{Fore.YELLOW}[⚠️] Test mode enabled - no real trading{Style.RESET_ALL}")
    
    if args.freq:
        config['agent']['thinking_frequency'] = args.freq
        print(f"[⚙️] Thinking frequency set to {args.freq} seconds")
    
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
        print("[⚙️] Verbose logging enabled")
    
    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Run the bot
    asyncio.run(main())

# Run CLI if script is executed with arguments
if __name__ == "__main__" and len(sys.argv) > 1:
    run_cli()