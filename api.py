from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime

def create_api_app(trading_bot):
    """Create Flask API for trading bot"""
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Get bot status"""
        return jsonify({
            'status': 'running' if trading_bot.is_running else 'stopped',
            'uptime': str(datetime.now() - trading_bot.start_time) if hasattr(trading_bot, 'start_time') else '0:00:00',
            'account': trading_bot.mt5_connector.get_account_info()
        })
    
    @app.route('/api/metrics', methods=['GET'])
    def get_metrics():
        """Get performance metrics"""
        metrics = trading_bot.performance_tracker.calculate_metrics()
        return jsonify(metrics)
    
    @app.route('/api/positions', methods=['GET'])
    def get_positions():
        """Get current positions"""
        positions = []
        for ticket, pos in trading_bot.positions.items():
            positions.append({
                'ticket': ticket,
                'symbol': pos['symbol'],
                'direction': 'BUY' if pos['direction'] == 1 else 'SELL',
                'entry_price': pos['entry_price'],
                'size': pos['size'],
                'unrealized_pnl': pos.get('unrealized_pnl', 0)
            })
        return jsonify(positions)
    
    @app.route('/api/trades', methods=['GET'])
    def get_trades():
        """Get trade history"""
        limit = request.args.get('limit', 50, type=int)
        trades = trading_bot.performance_tracker.get_recent_trades(limit)
        return jsonify(trades)
    
    @app.route('/api/predictions', methods=['GET'])
    def get_predictions():
        """Get latest predictions"""
        # Get latest predictions from each model
        predictions = {}
        for symbol in trading_bot.config['trading']['symbols']:
            predictions[symbol] = {
                'timestamp': datetime.now().isoformat(),
                'models': {}
            }
            # This would get actual predictions
        return jsonify(predictions)
    
    @app.route('/api/config', methods=['GET', 'POST'])
    def handle_config():
        """Get or update configuration"""
        if request.method == 'GET':
            # Return safe config (without sensitive data)
            safe_config = {
                'trading': trading_bot.config.get('trading', {}),
                'risk': trading_bot.config.get('risk', {}),
                'models': list(trading_bot.models.keys())
            }
            return jsonify(safe_config)
        
        elif request.method == 'POST':
            # Update configuration
            updates = request.json
            # Validate and apply updates
            # This would need proper validation
            return jsonify({'status': 'updated'})
    
    @app.route('/api/control/start', methods=['POST'])
    def start_bot():
        """Start trading bot"""
        if not trading_bot.is_running:
            # Start bot in background
            import asyncio
            asyncio.create_task(trading_bot.start())
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already running'})
    
    @app.route('/api/control/stop', methods=['POST'])
    def stop_bot():
        """Stop trading bot"""
        if trading_bot.is_running:
            import asyncio
            asyncio.create_task(trading_bot.stop())
            return jsonify({'status': 'stopped'})
        return jsonify({'status': 'already stopped'})
    
    @app.route('/api/control/close_all', methods=['POST'])
    def close_all_positions():
        """Close all positions"""
        import asyncio
        asyncio.create_task(trading_bot.close_all_positions())
        return jsonify({'status': 'closing all positions'})
    
    @app.errorhandler(Exception)
    def handle_error(error):
        """Handle API errors"""
        logging.error(f"API Error: {error}")
        return jsonify({'error': str(error)}), 500
    
    return app