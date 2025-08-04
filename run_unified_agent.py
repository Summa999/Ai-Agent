#!/usr/bin/env python3
"""
Unified AI Agent Launcher
Simple script to run the full AI agent
"""

import subprocess
import sys
import os

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           UNIFIED AI TRADING AGENT LAUNCHER               ║
    ╠═══════════════════════════════════════════════════════════╣
    ║                                                           ║
    ║  This will launch the AI Agent with full capabilities     ║
    ║  for both Forex and Cryptocurrency markets               ║
    ║                                                           ║
    ║  The agent will:                                          ║
    ║  • Think autonomously                                     ║
    ║  • Trade across multiple markets                          ║
    ║  • Learn from experience                                  ║
    ║  • Manage risk dynamically                                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not activated!")
        print("   Run: .\\venv\\Scripts\\activate (Windows)")
        print("   Or:  source venv/bin/activate (Linux/Mac)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Check configuration
    if not os.path.exists('config.yaml'):
        print("❌ config.yaml not found!")
        return
    
    print("\n🚀 Starting Unified AI Agent...\n")
    
    try:
        # Run the unified agent
        subprocess.run([sys.executable, 'main.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n✋ Agent stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running agent: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()