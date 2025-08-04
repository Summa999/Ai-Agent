import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import logging

def create_dashboard_app(trading_bot):
    """Create Dash web dashboard"""
    
    app = dash.Dash(__name__)
    
    # Define layout
    app.layout = html.Div([
        html.H1('AI Trading Bot Dashboard', style={'text-align': 'center'}),
        
        # Auto-refresh
        dcc.Interval(
            id='interval-component',
            interval=10*1000,  # Update every 10 seconds
            n_intervals=0
        ),
        
        # Top metrics
        html.Div([
            html.Div([
                html.H3('Account Balance'),
                html.H2(id='balance-display', children='\$0.00')
            ], className='metric-box', style={'width': '24%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3('Daily P&L'),
                html.H2(id='daily-pnl-display', children='\$0.00')
            ], className='metric-box', style={'width': '24%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3('Win Rate'),
                html.H2(id='win-rate-display', children='0.0%')
            ], className='metric-box', style={'width': '24%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3('Active Positions'),
                html.H2(id='positions-display', children='0')
            ], className='metric-box', style={'width': '24%', 'display': 'inline-block'}),
        ], style={'margin': '20px'}),
        
        # Charts
        html.Div([
            # Equity curve
            dcc.Graph(id='equity-curve'),
            
            # Position chart
            dcc.Graph(id='position-chart'),
            
            # Performance metrics
            dcc.Graph(id='performance-chart'),
            
            # Trade history table
            html.Div(id='trade-history-table')
        ])
    ])
    
    # Callbacks
    @app.callback(
        [Output('balance-display', 'children'),
         Output('daily-pnl-display', 'children'),
         Output('win-rate-display', 'children'),
         Output('positions-display', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_metrics(n):
        """Update dashboard metrics"""
        account_info = trading_bot.mt5_connector.get_account_info()
        performance = trading_bot.performance_tracker.calculate_metrics()
        
        balance = f"${account_info['balance']:.2f}" if account_info else "\$0.00"
        daily_pnl = f"${trading_bot.daily_pnl:.2f}"
        win_rate = f"{performance.get('win_rate', 0):.1%}"
        positions = str(len(trading_bot.positions))
        
        return balance, daily_pnl, win_rate, positions
    
    @app.callback(
        Output('equity-curve', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_equity_curve(n):
        """Update equity curve chart"""
        equity_data = trading_bot.performance_tracker.daily_equity
        
        if not equity_data:
            return go.Figure()
        
        df = pd.DataFrame(equity_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['balance'],
            mode='lines',
            name='Account Balance',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Time',
            yaxis_title='Balance ($)',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('position-chart', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_position_chart(n):
        """Update positions chart"""
        positions = trading_bot.positions
        
        if not positions:
            return go.Figure().add_annotation(
                text="No Active Positions",
                showarrow=False,
                font=dict(size=20)
            )
        
        # Create position data
        symbols = []
        pnl = []
        
        for pos in positions.values():
            symbols.append(pos['symbol'])
            pnl.append(pos.get('unrealized_pnl', 0))
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=pnl,
                marker_color=['green' if p > 0 else 'red' for p in pnl]
            )
        ])
        
        fig.update_layout(
            title='Active Positions P&L',
            xaxis_title='Symbol',
            yaxis_title='Unrealized P&L ($)',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('performance-chart', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_performance_chart(n):
        """Update performance metrics chart"""
        metrics = trading_bot.performance_tracker.calculate_metrics()
        
        if not metrics:
            return go.Figure()
        
        # Create radar chart
        categories = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 
                     'Avg Win/Loss', 'Recovery Factor']
        
        values = [
            metrics.get('win_rate', 0) * 100,
            min(metrics.get('profit_factor', 0) * 20, 100),
            min(metrics.get('sharpe_ratio', 0) * 33, 100),
            min(metrics.get('avg_win', 0) / abs(metrics.get('avg_loss', 1)) * 50, 100),
            min(metrics.get('recovery_factor', 0) * 10, 100)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Performance Metrics"
        )
        
        return fig
    
    @app.callback(
        Output('trade-history-table', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_trade_history(n):
        """Update trade history table"""
        trades = trading_bot.performance_tracker.get_recent_trades(20)
        
        if not trades:
            return html.Div("No recent trades")
        
        df = pd.DataFrame(trades)
        
        # Format for display
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Select columns to display
        display_columns = ['symbol', 'direction', 'entry_time', 'entry_price', 
                          'exit_time', 'exit_price', 'profit']
        
        available_columns = [col for col in display_columns if col in df.columns]
        
        return html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in available_columns])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(df.iloc[i][col]) for col in available_columns
                ]) for i in range(len(df))
            ])
        ], style={'width': '100%', 'text-align': 'center'})
    
    # Add CSS styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                }
                .metric-box {
                    background-color: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .metric-box h3 {
                    color: #666;
                    margin: 0;
                }
                .metric-box h2 {
                    color: #333;
                    margin: 10px 0;
                }
                table {
                    border-collapse: collapse;
                    margin: 20px auto;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                }
                tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    return app