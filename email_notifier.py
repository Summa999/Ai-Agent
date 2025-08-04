import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
import logging
from datetime import datetime

class EmailNotifier:
    def __init__(self, config: Dict):
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email')
        self.sender_password = config.get('sender_password')
        self.recipient_emails = config.get('recipient_emails', [])
        self.logger = logging.getLogger(__name__)
    
    def send_email(self, subject: str, body: str, html_body: str = None):
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            
            # Add text part
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
    
    def send_daily_report(self, metrics: Dict):
        """Send daily performance report via email"""
        subject = f"Trading Bot Daily Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Plain text version
        body = f"""
Trading Bot Daily Performance Report

Date: {datetime.now().strftime('%Y-%m-%d')}

Performance Summary:
- Total P&L: ${metrics.get('total_profit', 0):.2f}
- Total Trades: {metrics.get('total_trades', 0)}
- Win Rate: {metrics.get('win_rate', 0):.1%}
- Profit Factor: {metrics.get('profit_factor', 0):.2f}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.1%}

Best Day: ${metrics.get('best_day', 0):.2f}
Worst Day: ${metrics.get('worst_day', 0):.2f}
Average Daily P&L: ${metrics.get('avg_daily_pnl', 0):.2f}

This is an automated report from your trading bot.
"""
        
        # HTML version
        html_body = self._create_html_report(metrics)
        
        self.send_email(subject, body, html_body)
    
    def _create_html_report(self, metrics: Dict) -> str:
        """Create HTML formatted report"""
        pnl_color = "green" if metrics.get('total_profit', 0) >= 0 else "red"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        th {{ background-color: #4CAF50; color: white; }}
        .profit {{ color: green; font-weight: bold; }}
        .loss {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h2>Trading Bot Daily Performance Report</h2>
    <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h3>Performance Summary</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Total P&L</td>
            <td class="{pnl_color}">${metrics.get('total_profit', 0):.2f}</td>
        </tr>
        <tr>
            <td>Total Trades</td>
            <td>{metrics.get('total_trades', 0)}</td>
        <tr>
            <td>Win Rate</td>
            <td>{metrics.get('win_rate', 0):.1%}</td>
        </tr>
        <tr>
            <td>Profit Factor</td>
            <td>{metrics.get('profit_factor', 0):.2f}</td>
        </tr>
        <tr>
            <td>Sharpe Ratio</td>
            <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
        </tr>
        <tr>
            <td>Max Drawdown</td>
            <td>{metrics.get('max_drawdown', 0):.1%}</td>
        </tr>
    </table>
    
    <h3>Daily Statistics</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Best Day</td>
            <td class="profit">${metrics.get('best_day', 0):.2f}</td>
        </tr>
        <tr>
            <td>Worst Day</td>
            <td class="loss">${metrics.get('worst_day', 0):.2f}</td>
        </tr>
        <tr>
            <td>Average Daily P&L</td>
            <td>${metrics.get('avg_daily_pnl', 0):.2f}</td>
        </tr>
    </table>
    
    <p><i>This is an automated report from your AI trading bot.</i></p>
</body>
</html>
"""
        return html
    
    def send_alert(self, alert_type: str, message: str):
        """Send alert email"""
        subject = f"Trading Bot Alert - {alert_type.upper()}"
        
        body = f"""
Trading Bot Alert

Type: {alert_type.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}

Please check your trading system.
"""
        
        self.send_email(subject, body)