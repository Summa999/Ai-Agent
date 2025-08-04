import asyncio
import aiohttp
from typing import Dict, List
import logging
from datetime import datetime

class TelegramNotifier:
    def __init__(self, config: Dict):
        self.bot_token = config.get('bot_token')
        self.chat_id = config.get('chat_id')
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: str, parse_mode: str = 'HTML'):
        """Send text message"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to send message: {await response.text()}")
                    
        except Exception as e:
            self.logger.error(f"Telegram send error: {e}")
    
    async def send_trade_notification(self, trade: Dict):
        """Send trade execution notification"""
        emoji = "📈" if trade['type'] == 'BUY' else "📉"
        
        message = f"""
{emoji} <b>Trade Executed</b>

Symbol: <code>{trade['symbol']}</code>
Type: <b>{trade['type']}</b>
Price: <code>{trade['price']:.5f}</code>
Size: <code>{trade['size']:.2f}</code>
Confidence: <b>{trade['confidence']:.1%}</b>
Regime: <i>{trade['regime']}</i>

Time: {datetime.now().strftime('%H:%M:%S')}
"""
        
        await self.send_message(message)
    
    async def send_daily_report(self, metrics: Dict):
        """Send daily performance report"""
        pnl_emoji = "✅" if metrics['total_profit'] >= 0 else "❌"
        
        message = f"""
📊 <b>Daily Performance Report</b>

{pnl_emoji} P&L: <b>${metrics['total_profit']:.2f}</b>
📈 Total Trades: <code>{metrics['total_trades']}</code>
✅ Win Rate: <b>{metrics['win_rate']:.1%}</b>
💰 Profit Factor: <code>{metrics['profit_factor']:.2f}</code>

📉 Max Drawdown: <code>{metrics['max_drawdown']:.1%}</code>
📊 Sharpe Ratio: <code>{metrics['sharpe_ratio']:.2f}</code>
⚡ Avg Trade Duration: <code>{metrics['avg_duration_hours']:.1f}h</code>

Best Performing:
{self._format_symbol_performance(metrics.get('symbol_performance', {}))}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        await self.send_message(message)
    
    def _format_symbol_performance(self, symbol_perf: Dict) -> str:
        """Format symbol performance for display"""
        if not symbol_perf:
            return "No data available"
        
        lines = []
        for symbol, data in symbol_perf.items():
            profit = data.get('profit', {}).get('sum', 0)
            count = data.get('profit', {}).get('count', 0)
            emoji = "📈" if profit > 0 else "📉"
            lines.append(f"{emoji} {symbol}: ${profit:.2f} ({count} trades)")
        
        return "\n".join(lines[:5])  # Top 5 symbols
    
    async def send_alert(self, alert_type: str, message: str):
        """Send alert notification"""
        emoji_map = {
            'error': '❗',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅',
            'critical': '🚨'
        }
        
        emoji = emoji_map.get(alert_type, '📢')
        
        alert_message = f"""
{emoji} <b>Alert: {alert_type.upper()}</b>

{message}

Time: {datetime.now().strftime('%H:%M:%S')}
"""
        
        await self.send_message(alert_message)
    
    async def send_market_update(self, market_data: Dict):
        """Send market condition update"""
        message = f"""
🌍 <b>Market Update</b>

Regime: <b>{market_data['regime']['regime']}</b>
Volatility: <code>{market_data['volatility']:.1%}</code>
Trend: <i>{market_data['trend']}</i>

Top Opportunities:
{self._format_opportunities(market_data.get('opportunities', []))}

Sentiment:
• News: {self._sentiment_emoji(market_data['sentiment']['news'])}
• Social: {self._sentiment_emoji(market_data['sentiment']['social'])}
• Overall: {self._sentiment_emoji(market_data['sentiment']['overall'])}
"""
        
        await self.send_message(message)
    
    def _format_opportunities(self, opportunities: List[Dict]) -> str:
        """Format trading opportunities"""
        if not opportunities:
            return "No strong opportunities detected"
        
        lines = []
        for opp in opportunities[:3]:
            direction = "🔼" if opp['direction'] == 'BUY' else "🔽"
            lines.append(
                f"{direction} {opp['symbol']} - "
                f"Confidence: {opp['confidence']:.1%}"
            )
        
        return "\n".join(lines)
    
    def _sentiment_emoji(self, sentiment: float) -> str:
        """Convert sentiment score to emoji"""
        if sentiment > 0.7:
            return "😄 Very Positive"
        elif sentiment > 0.5:
            return "🙂 Positive"
        elif sentiment > 0.3:
            return "😐 Neutral"
        elif sentiment > 0.1:
            return "😕 Negative"
        else:
            return "😟 Very Negative"