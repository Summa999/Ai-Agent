# integration_script_fixed.py - Fixed version for Windows encoding
import os
import shutil
import yaml

def setup_enhanced_agent():
    """Setup enhanced AI agent with detailed logging - Windows compatible"""
    
    print("🚀 Setting up Enhanced AI Agent...")
    
    # 1. Install required packages
    print("📦 Installing required packages...")
    os.system("pip install colorama")
    
    # 2. Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("📁 Created logs directory")
    
    # 3. Update config with enhanced settings
    print("⚙️ Updating configuration...")
    
    enhanced_config = {
        'trading': {
            'enabled': False,  # Keep false for observation
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        },
        'agent': {
            'thinking_frequency': 30,  # Think every 30 seconds
            'detailed_analysis': True,
            'verbose_reasoning': True,
            'decision_logging': True,
            'learning_tracking': True
        },
        'logging': {
            'level': 'DEBUG',
            'console_colors': True,
            'show_reasoning': True,
            'show_analysis': True,
            'show_learning': True
        },
        'display': {
            'real_time_updates': True,
            'show_market_data': True,
            'show_indicators': True,
            'show_opportunities': True
        }
    }
    
    # Load existing config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except:
        config = {}
    
    # Merge enhanced settings
    for key, value in enhanced_config.items():
        if key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration updated!")
    
    # 4. Create enhanced main.py with Windows-compatible encoding
    print("🔧 Creating enhanced main script...")
    
    enhanced_main = '''#!/usr/bin/env python3
"""
Enhanced AI Trading Agent with Real-time Insights
"""
import asyncio
import sys
import os
import yaml
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced components
from enhanced_logging_config import setup_enhanced_logging, AIDecisionLogger
from enhanced_ai_agent import EnhancedAIAgent

# Import existing components
from connectors.mt5_connector import MT5Connector
from connectors.crypto_connector import CryptoConnector

class EnhancedTradingBot:
    """Enhanced trading bot with detailed insights"""
    
    def __init__(self):
        self.config = self.load_config()
        self.markets = {}
        self.enhanced_agent = None
        
        # Setup enhanced logging
        setup_enhanced_logging()
        self.logger = logging.getLogger('EnhancedBot')
        self.decision_logger = AIDecisionLogger()
    
    def load_config(self):
        """Load configuration"""
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    async def initialize(self):
        """Initialize enhanced bot"""
        self.logger.info("[ROBOT] ENHANCED AI TRADING AGENT STARTING")
        self.logger.info("=" * 80)
        self.logger.info("[TARGET] Features Active:")
        self.logger.info("  • Real-time decision reasoning")
        self.logger.info("  • Deep technical analysis")
        self.logger.info("  • Market sentiment tracking")
        self.logger.info("  • Advanced learning algorithms")
        self.logger.info("  • Risk management optimization")
        self.logger.info("=" * 80)
        
        # Initialize market connectors
        if self.config.get('markets', {}).get('forex', {}).get('enabled'):
            self.markets['forex'] = MT5Connector(self.config['mt5'])
            
        if self.config.get('markets', {}).get('crypto', {}).get('enabled'):
            self.markets['crypto'] = CryptoConnector(self.config['markets']['crypto'])
        
        # Connect to markets
        connected_markets = []
        for market_name, connector in self.markets.items():
            self.logger.info(f"[PLUG] Connecting to {market_name.upper()}...")
            
            if await connector.connect():
                connected_markets.append(market_name)
                account_info = await connector.get_account_info()
                
                self.logger.info(f"[CHECK] {market_name.upper()} connected")
                self.logger.info(f"  • Account: {account_info.get('account', 'N/A')}")
                self.logger.info(f"  • Balance: ${account_info.get('balance', 0):.2f}")
                
        if not connected_markets:
            self.logger.error("[X] No markets connected!")
            return False
        
        # Initialize enhanced AI agent
        self.enhanced_agent = EnhancedAIAgent(self.config, self.markets)
        
        self.logger.info(f"[BRAIN] Enhanced AI Agent initialized")
        self.logger.info(f"[LIGHTNING] Thinking frequency: {self.enhanced_agent.thinking_freq}s")
        
        return True
    
    async def run(self):
        """Run enhanced trading bot"""
        if not await self.initialize():
            return
        
        self.logger.info("[ROCKET] ENHANCED AI AGENT IS NOW ACTIVE")
        self.logger.info("[BRAIN] Real-time analysis and decision tracking enabled")
        self.logger.info("[STOP] Press Ctrl+C to stop")
        self.logger.info("=" * 80)
        
        try:
            # Start enhanced thinking loop
            await self.enhanced_agent.start_thinking_loop()
            
        except KeyboardInterrupt:
            self.logger.info("[STOP] Enhanced AI Agent stopped by user")
        except Exception as e:
            self.logger.error(f"[X] Error: {e}")
        finally:
            # Cleanup
            for connector in self.markets.values():
                await connector.disconnect()
            
            self.logger.info("[WAVE] Enhanced AI Agent shutdown complete")

async def main():
    """Main entry point"""
    bot = EnhancedTradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n[WAVE] Goodbye!")
'''
    
    # Write with UTF-8 encoding to handle special characters
    with open('enhanced_main.py', 'w', encoding='utf-8') as f:
        f.write(enhanced_main)
    
    print("✅ Enhanced main script created!")
    
    # 5. Create enhanced logging config
    print("🔧 Creating enhanced logging configuration...")
    
    logging_config = '''# enhanced_logging_config.py - Enhanced logging with Windows compatibility
import logging
import os
from datetime import datetime
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for Windows compatibility"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    
    def format(self, record):
        # Add color to log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Format message
        formatted = super().format(record)
        
        # Add special formatting for AI messages
        if '[ROBOT]' in formatted:
            formatted = f"{Fore.MAGENTA}{formatted}{Style.RESET_ALL}"
        elif '[BRAIN]' in formatted:
            formatted = f"{Fore.BLUE}{formatted}{Style.RESET_ALL}"
        elif '[TARGET]' in formatted:
            formatted = f"{Fore.GREEN}{formatted}{Style.RESET_ALL}"
        elif '[LIGHTNING]' in formatted:
            formatted = f"{Fore.YELLOW}{formatted}{Style.RESET_ALL}"
        elif '[CHECK]' in formatted:
            formatted = f"{Fore.GREEN}{formatted}{Style.RESET_ALL}"
        elif '[X]' in formatted:
            formatted = f"{Fore.RED}{formatted}{Style.RESET_ALL}"
        
        return formatted

def setup_enhanced_logging():
    """Setup enhanced logging configuration"""
    
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for detailed logs
    log_filename = f"logs/ai_agent_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Specialized loggers
    trading_logger = logging.getLogger('TradingDecisions')
    trading_handler = logging.FileHandler(f"logs/trading_decisions_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
    trading_handler.setFormatter(file_formatter)
    trading_logger.addHandler(trading_handler)
    
    print(f"[CHECK] Enhanced logging setup complete")
    print(f"[FILE] Logs will be saved to: {log_filename}")

class AIDecisionLogger:
    """Specialized logger for AI decision tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger('AIDecisions')
        self.decisions_count = 0
    
    def log_decision(self, symbol, action, reasoning, confidence):
        """Log AI trading decision"""
        self.decisions_count += 1
        
        self.logger.info(f"DECISION #{self.decisions_count}")
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Action: {action}")
        self.logger.info(f"Confidence: {confidence:.1f}%")
        self.logger.info(f"Reasoning: {reasoning}")
        self.logger.info("-" * 50)
'''
    
    with open('enhanced_logging_config.py', 'w', encoding='utf-8') as f:
        f.write(logging_config)
    
    print("✅ Enhanced logging configuration created!")
    
    # 6. Installation complete
    print("🎉 ENHANCED AI AGENT SETUP COMPLETE!")
    print()
    print("🚀 To start with enhanced features:")
    print("   python enhanced_main.py")
    print()
    print("📊 Features enabled:")
    print("   • 30-second thinking cycles")
    print("   • Detailed decision reasoning")
    print("   • Real-time market analysis")
    print("   • Technical indicator insights")
    print("   • Learning progress tracking")
    print("   • Colored console output (Windows compatible)")
    print()
    print("⚙️ Configuration updated in config.yaml")
    print("📝 Detailed logs saved to logs/ directory")
    print("🎨 Windows-compatible colored output enabled")

if __name__ == "__main__":
    setup_enhanced_agent()