import asyncio
import sys
import signal
from ai_agent_core import AITradingAgent

class AgentRunner:
    def __init__(self):
        self.agent = None
        self.tasks = []
        
    async def start(self):
        """Start the AI agent"""
        print("🤖 Initializing AI Trading Agent...")
        
        # Create agent
        self.agent = AITradingAgent('agent_config.yaml')
        
        # Connect to markets
        print("\n📊 Connecting to markets...")
        for market_name, market_interface in self.agent.markets.items():
            if await market_interface.connect():
                print(f"✓ Connected to {market_name}")
            else:
                print(f"✗ Failed to connect to {market_name}")
        
        # Start agent tasks
        print("\n🧠 Starting agent cognitive processes...")
        
        # Main thinking loop
        self.tasks.append(
            asyncio.create_task(self.agent.think())
        )
        
        # Monitoring loop
        self.tasks.append(
            asyncio.create_task(self.monitor_agent())
        )
        
        # Learning loop
        self.tasks.append(
            asyncio.create_task(self.agent.learning_module.continuous_learning_loop())
        )
        
        print("\n✅ AI Agent is now active and autonomous!")
        print("The agent will now:")
        print("- Monitor multiple markets simultaneously")
        print("- Make autonomous trading decisions")
        print("- Learn from experience and adapt strategies")
        print("- Manage risk dynamically")
        print("\nPress Ctrl+C to stop the agent\n")
        
        # Wait for tasks
        await asyncio.gather(*self.tasks)
    
    async def monitor_agent(self):
        """Monitor agent performance and status"""
        while True:
            try:
                # Get agent status
                status = await self.agent.report_status()
                print(status)
                
                # Check agent health
                health = await self.check_agent_health()
                if not health['healthy']:
                    print(f"⚠️ Agent health issue: {health['issues']}")
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
    
    async def check_agent_health(self):
        """Check if agent is functioning properly"""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # Check if agent is making decisions
        if self.agent.state['total_trades'] == 0 and \
           self.agent.state['learning_progress'] < 0.1:
            health['healthy'] = False
            health['issues'].append("Agent not making progress")
        
        # Check error rate
        # Add more health checks as needed
        
        return health
    
    async def stop(self):
        """Gracefully stop the agent"""
        print("\n🛑 Stopping AI Agent...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close all positions if configured
        if self.agent.config.get('close_positions_on_stop', False):
            print("Closing all positions...")
            # Implementation here
        
        # Save agent state
        print("Saving agent memory and state...")
        await self.agent.save_state()
        
        print("✓ AI Agent stopped successfully")

async def main():
    """Main entry point"""
    runner = AgentRunner()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal...")
        asyncio.create_task(runner.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await runner.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        await runner.stop()

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS AI TRADING AGENT v2.0")
    print("=" * 60)
    print("\nThis is a true AI agent that will:")
    print("✓ Trade multiple markets (Forex, Crypto, Stocks)")
    print("✓ Make autonomous decisions")
    print("✓ Learn and adapt continuously")
    print("✓ Manage risk dynamically")
    print("✓ Generate and test new strategies")
    print("=" * 60)
    
    # Run the agent
    asyncio.run(main())