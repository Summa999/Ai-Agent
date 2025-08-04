# integration_script.py - Easy setup for enhanced AI agent
import os
import shutil
import yaml

def setup_enhanced_agent():
    """Setup enhanced AI agent with detailed logging"""
    
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
    
    # 4. Create enhanced main.py
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
        self.logger.info("🤖 ENHANCED AI TRADING AGENT STARTING")
        self.logger.info("=" * 80)
        self.logger.info("🎯 Features Active:")
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
            self.logger.info(f"🔌 Connecting to {market_name.upper()}...")
            
            if await connector.connect():
                connected_markets.append(market_name)
                account_info = await connector.get_account_info()
                
                self.logger.info(f"✅ {market_name.upper()} connected")
                self.logger.info(f"  • Account: {account_info.get('account', 'N/A')}")
                self.logger.info(f"  • Balance: ${account_info.get('balance', 0):.2f}")
                
        if not connected_markets:
            self.logger.error("❌ No markets connected!")
            return False
        
        # Initialize enhanced AI agent
        self.enhanced_agent = EnhancedAIAgent(self.config, self.markets)
        
        self.logger.info(f"🧠 Enhanced AI Agent initialized")
        self.logger.info(f"⚡ Thinking frequency: {self.enhanced_agent.thinking_freq}s")
        
        return True
    
    async def run(self):
        """Run enhanced trading bot"""
        if not await self.initialize():
            return
        
        self.logger.info("🚀 ENHANCED AI AGENT IS NOW ACTIVE")
        self.logger.info("🧠 Real-time analysis and decision tracking enabled")
        self.logger.info("⏹️ Press Ctrl+C to stop")
        self.logger.info("=" * 80)
        
        try:
            # Start enhanced thinking loop
            await self.enhanced_agent.start_thinking_loop()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Enhanced AI Agent stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
        finally:
            # Cleanup
            for connector in self.markets.values():
                await connector.disconnect()
            
            self.logger.info("👋 Enhanced AI Agent shutdown complete")

async def main():
    """Main entry point"""
    bot = EnhancedTradingBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
'''
    
    with open('enhanced_main.py', 'w') as f:
        f.write(enhanced_main)
    
    print("✅ Enhanced main script created!")
    
    # 5. Installation complete
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
    print("   • Colored console output")
    print()
    print("⚙️ Configuration updated in config.yaml")
    print("📝 Detailed logs saved to logs/ directory")

if __name__ == "__main__":
    setup_enhanced_agent()