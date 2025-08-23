"""
Backtesting Engine for Alpha Factor Strategies
Implements realistic portfolio simulation with transaction costs and constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Professional backtesting engine for quantitative trading strategies
    """
    
    def __init__(self, config=None):
        """
        Initialize backtest engine with configuration
        
        Parameters:
        config (dict): Backtesting configuration parameters
        """
        default_config = {
            'initial_capital': 100000,
            'transaction_cost': 0.001,  # 0.1% per trade
            'max_leverage': 1.0,  # No leverage
            'rebalance_frequency': 'daily',
            'signal_threshold': 0.005,  # 0.5% threshold for signals
            'stop_loss': None,  # No stop loss by default
            'take_profit': None,  # No take profit by default
            'min_holding_period': 1,  # Minimum days to hold position
            'max_position_size': 1.0,  # 100% of capital
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize results storage
        self.results = []
        self.trades = []
        self.metrics = {}
        
    def run(self, predictions_df, benchmark_data=None):
        """
        Run the backtest simulation
        
        Parameters:
        predictions_df (pd.DataFrame): DataFrame with predictions and actual returns
        benchmark_data (pd.DataFrame): Optional benchmark data (defaults to buy-and-hold)
        
        Returns:
        dict: Backtest results including performance metrics
        """
        print(f'Starting backtest with {len(predictions_df)} predictions')
        
        # Ensure data is sorted by date
        predictions_df = predictions_df.sort_values('date').reset_index(drop=True)
        
        # Initialize portfolio state
        portfolio_value = self.config['initial_capital']
        cash = portfolio_value
        position = 0  # -1: short, 0: flat, 1: long
        position_size = 0  # Dollar amount of position
        entry_price = 0
        entry_date = None
        holding_period = 0
        
        # Benchmark tracking (buy-and-hold strategy)
        benchmark_value = self.config['initial_capital']
        benchmark_shares = self.config['initial_capital'] / predictions_df.iloc[0]['price'] if len(predictions_df) > 0 else 0
        
        # Risk management
        consecutive_losses = 0
        max_consecutive_losses = 0
        total_trades = 0
        winning_trades = 0
        
        # Process each trading day
        for i, row in predictions_df.iterrows():
            if i == 0:
                # Initialize first day
                self.results.append({
                    'date': row['date'],
                    'portfolio_value': portfolio_value,
                    'benchmark_value': benchmark_value,
                    'position': position,
                    'signal': 0,
                    'prediction': row.get('predicted_return', 0),
                    'actual': row.get('actual_return', 0),
                    'cash': cash,
                    'position_size': position_size,
                    'pnl': 0,
                    'cumulative_return': 0,
                    'benchmark_return': 0,
                    'price': row['price']
                })
                continue
            
            prev_result = self.results[i - 1]
            actual_return = row.get('actual_return', 0)
            predicted_return = row.get('predicted_return', 0)
            current_price = row['price']
            
            # Update holding period
            if position != 0:
                holding_period += 1
            
            # Calculate P&L from existing position
            pnl = 0
            if position != 0 and position_size > 0:
                position_return = position * actual_return
                pnl = position_size * position_return
                portfolio_value += pnl
                
                # Update position value
                position_size *= (1 + position_return)
            
            # Generate trading signal
            signal = 0
            if abs(predicted_return) > self.config['signal_threshold']:
                if predicted_return > self.config['signal_threshold']:
                    signal = 1  # Long signal
                elif predicted_return < -self.config['signal_threshold']:
                    signal = -1  # Short signal
            
            # Risk management: reduce position size after consecutive losses
            risk_adjustment = 1.0
            if consecutive_losses > 2:
                risk_adjustment = max(0.5, 1.0 - (consecutive_losses - 2) * 0.1)
            
            # Execute trades
            can_trade = (position == 0) or (holding_period >= self.config['min_holding_period'])
            if signal != position and can_trade:
                # Close existing position
                if position != 0:
                    trade_return = position * actual_return
                    trade_pnl = position_size * trade_return
                    transaction_cost = position_size * self.config['transaction_cost']
                    
                    cash += position_size - transaction_cost
                    portfolio_value = cash
                    
                    # Record trade
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': row['date'],
                        'direction': position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position_size,
                        'pnl': trade_pnl - transaction_cost,
                        'return': trade_return,
                        'holding_period': holding_period,
                        'prediction': predicted_return,
                        'actual': actual_return
                    })
                    
                    total_trades += 1
                    if trade_pnl > 0:
                        winning_trades += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    position = 0
                    position_size = 0
                    holding_period = 0
                
                # Open new position
                if signal != 0:
                    available_capital = cash * self.config['max_position_size'] * risk_adjustment
                    transaction_cost = available_capital * self.config['transaction_cost']
                    
                    position_size = available_capital - transaction_cost
                    cash -= available_capital
                    position = signal
                    entry_price = current_price
                    entry_date = row['date']
                    holding_period = 1
            
            # Update benchmark (buy-and-hold)
            benchmark_value = benchmark_shares * current_price
            
            # Calculate returns
            cumulative_return = (portfolio_value - self.config['initial_capital']) / self.config['initial_capital']
            benchmark_return = (benchmark_value - self.config['initial_capital']) / self.config['initial_capital']
            
            # Store results
            self.results.append({
                'date': row['date'],
                'portfolio_value': round(portfolio_value, 2),
                'benchmark_value': round(benchmark_value, 2),
                'position': position,
                'signal': signal,
                'prediction': predicted_return,
                'actual': actual_return,
                'cash': round(cash, 2),
                'position_size': round(position_size, 2),
                'pnl': round(pnl, 2),
                'cumulative_return': cumulative_return,
                'benchmark_return': benchmark_return,
                'consecutive_losses': consecutive_losses,
                'price': current_price
            })
        
        # Calculate comprehensive performance metrics
        self._calculate_metrics()
        
        # Print summary
        print(f'Backtest complete: {len(self.results)} days, {total_trades} trades')
        if total_trades > 0:
            print(f'Win rate: {(winning_trades / total_trades) * 100:.1f}%')
        
        return {
            'results': pd.DataFrame(self.results),
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'metrics': self.metrics
        }
    
    def _calculate_metrics(self):
        """
        Calculate comprehensive performance metrics
        """
        if len(self.results) < 2:
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Calculate daily returns
        portfolio_values = results_df['portfolio_value'].values
        benchmark_values = results_df['benchmark_value'].values
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        final_result = results_df.iloc[-1]
        total_return = final_result['cumulative_return']
        benchmark_total_return = final_result['benchmark_return']
        
        # Annualized returns
        trading_days = len(results_df)
        years_elapsed = trading_days / 252
        
        if years_elapsed > 0:
            annualized_return = (1 + total_return) ** (1 / years_elapsed) - 1
            benchmark_annualized_return = (1 + benchmark_total_return) ** (1 / years_elapsed) - 1
        else:
            annualized_return = total_return
            benchmark_annualized_return = benchmark_total_return
        
        # Volatility
        if len(returns) > 1:
            returns_std = np.std(returns, ddof=1)
            volatility = returns_std * np.sqrt(252)  # Annualized
        else:
            volatility = 0
        
        # Sharpe ratio
        excess_return = annualized_return - self.config['risk_free_rate']
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        current_drawdown = 0
        drawdown_duration = 0
        max_drawdown_duration = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
                drawdown_duration = 0
            else:
                drawdown_duration += 1
                max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
            
            current_drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Beta calculation
        beta = 1.0
        if len(returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns, ddof=1)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
        
        # Alpha (Jensen's alpha)
        alpha = annualized_return - (self.config['risk_free_rate'] + 
                                   beta * (benchmark_annualized_return - self.config['risk_free_rate']))
        
        # Information ratio
        tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
        information_ratio = ((annualized_return - benchmark_annualized_return) / tracking_error 
                           if tracking_error > 0 else 0)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            winning_trades_df = trades_df[trades_df['pnl'] > 0]
            losing_trades_df = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades_df) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades_df['pnl'].mean() if len(winning_trades_df) > 0 else 0
            avg_loss = abs(losing_trades_df['pnl'].mean()) if len(losing_trades_df) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            avg_holding_period = trades_df['holding_period'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_period = 0
        
        # Prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for result in self.results:
            pred = result['prediction']
            actual = result['actual']
            
            if abs(pred) > self.config['signal_threshold'] or abs(actual) > self.config['signal_threshold']:
                total_predictions += 1
                if (pred > 0 and actual > 0) or (pred < 0 and actual < 0):
                    correct_predictions += 1
        
        prediction_accuracy = (correct_predictions / total_predictions * 100 
                             if total_predictions > 0 else 0)
        
        # Store all metrics
        self.metrics = {
            # Return metrics
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'benchmark_total_return': benchmark_total_return * 100,
            'benchmark_annualized_return': benchmark_annualized_return * 100,
            'outperformance': (annualized_return - benchmark_annualized_return) * 100,
            
            # Risk metrics
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'max_drawdown_duration': max_drawdown_duration,
            'beta': beta,
            'alpha': alpha * 100,
            'information_ratio': information_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Trade metrics
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in self.trades if t['pnl'] < 0]),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            
            # Model metrics
            'prediction_accuracy': prediction_accuracy,
            
            # Portfolio metrics
            'final_value': final_result['portfolio_value'],
            'max_consecutive_losses': max([
                max([0] + [len(list(group)) for k, group in 
                          __import__('itertools').groupby([t['pnl'] < 0 for t in self.trades]) if k])
            ]) if self.trades else 0
        }
    
    def _calculate_tracking_error(self, returns, benchmark_returns):
        """
        Calculate tracking error (standard deviation of excess returns)
        """
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return 0
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)  # Annualized
        
        return tracking_error
    
    def get_trade_analysis(self):
        """
        Get detailed trade analysis
        """
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        long_trades = trades_df[trades_df['direction'] == 1]
        short_trades = trades_df[trades_df['direction'] == -1]
        
        def analyze_trades(trades_subset, direction_name):
            if len(trades_subset) == 0:
                return {
                    'count': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'total_pnl': 0
                }
            
            winning_trades = trades_subset[trades_subset['pnl'] > 0]
            return {
                'count': len(trades_subset),
                'win_rate': len(winning_trades) / len(trades_subset) * 100,
                'avg_return': trades_subset['return'].mean() * 100,
                'total_pnl': trades_subset['pnl'].sum()
            }
        
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()] if len(trades_df) > 0 else None
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()] if len(trades_df) > 0 else None
        
        return {
            'total': len(trades_df),
            'long': analyze_trades(long_trades, 'Long'),
            'short': analyze_trades(short_trades, 'Short'),
            'avg_holding_period': trades_df['holding_period'].mean(),
            'best_trade': {
                'date': best_trade['exit_date'],
                'direction': 'Long' if best_trade['direction'] == 1 else 'Short',
                'pnl': best_trade['pnl'],
                'return': best_trade['return'] * 100
            } if best_trade is not None else None,
            'worst_trade': {
                'date': worst_trade['exit_date'],
                'direction': 'Long' if worst_trade['direction'] == 1 else 'Short',
                'pnl': worst_trade['pnl'],
                'return': worst_trade['return'] * 100
            } if worst_trade is not None else None
        }
    
    def export_to_csv(self, filename='backtest_results.csv'):
        """
        Export results to CSV format
        """
        if not self.results:
            print("No results to export")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Format columns for export
        export_df = results_df.copy()
        export_df['prediction_pct'] = (export_df['prediction'] * 100).round(4)
        export_df['actual_pct'] = (export_df['actual'] * 100).round(4)
        export_df['cumulative_return_pct'] = (export_df['cumulative_return'] * 100).round(4)
        export_df['benchmark_return_pct'] = (export_df['benchmark_return'] * 100).round(4)
        
        # Select columns for export
        export_columns = [
            'date', 'portfolio_value', 'benchmark_value', 'position', 'signal',
            'prediction_pct', 'actual_pct', 'pnl', 'cumulative_return_pct', 'benchmark_return_pct'
        ]
        
        export_df[export_columns].to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def plot_performance(self, save_plot=False, filename='backtest_performance.png'):
        """
        Create performance visualization (requires matplotlib)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
        except ImportError:
            print("matplotlib not available for plotting")
            return
        
        if not self.results:
            print("No results to plot")
            return
        
        results_df = pd.DataFrame(self.results)
        results_df['date'] = pd.to_datetime(results_df['date'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Portfolio value over time
        ax1.plot(results_df['date'], results_df['portfolio_value'], 
                label='Strategy', linewidth=2, color='blue')
        ax1.plot(results_df['date'], results_df['benchmark_value'], 
                label='Benchmark (Buy & Hold)', linewidth=2, color='red', alpha=0.7)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Performance Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax2.plot(results_df['date'], results_df['cumulative_return'] * 100, 
                label='Strategy', linewidth=2, color='blue')
        ax2.plot(results_df['date'], results_df['benchmark_return'] * 100, 
                label='Benchmark', linewidth=2, color='red', alpha=0.7)
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.set_title('Cumulative Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Position and signals
        ax3.fill_between(results_df['date'], 0, results_df['position'], 
                        alpha=0.3, label='Position (1=Long, -1=Short)')
        ax3.plot(results_df['date'], results_df['signal'], 
                'o', markersize=2, alpha=0.6, label='Signals')
        ax3.set_ylabel('Position/Signal')
        ax3.set_xlabel('Date')
        ax3.set_title('Trading Positions and Signals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        else:
            plt.show()
    
    def get_performance_summary(self):
        """
        Get formatted performance summary
        """
        if not self.metrics:
            return "No metrics calculated"
        
        summary = f"""
BACKTEST PERFORMANCE SUMMARY
{'=' * 50}

RETURN METRICS:
  Total Return:              {self.metrics['total_return']:8.2f}%
  Annualized Return:         {self.metrics['annualized_return']:8.2f}%
  Benchmark Return:          {self.metrics['benchmark_annualized_return']:8.2f}%
  Outperformance:            {self.metrics['outperformance']:8.2f}%

RISK METRICS:
  Volatility:                {self.metrics['volatility']:8.2f}%
  Sharpe Ratio:              {self.metrics['sharpe_ratio']:8.2f}
  Maximum Drawdown:          {self.metrics['max_drawdown']:8.2f}%
  Max Drawdown Duration:     {self.metrics['max_drawdown_duration']:8.0f} days
  Beta:                      {self.metrics['beta']:8.2f}
  Alpha:                     {self.metrics['alpha']:8.2f}%
  Information Ratio:         {self.metrics['information_ratio']:8.2f}
  Calmar Ratio:              {self.metrics['calmar_ratio']:8.2f}

TRADING METRICS:
  Total Trades:              {self.metrics['total_trades']:8.0f}
  Winning Trades:            {self.metrics['winning_trades']:8.0f}
  Losing Trades:             {self.metrics['losing_trades']:8.0f}
  Win Rate:                  {self.metrics['win_rate']:8.1f}%
  Profit Factor:             {self.metrics['profit_factor']:8.2f}
  Avg Holding Period:        {self.metrics['avg_holding_period']:8.1f} days

MODEL METRICS:
  Prediction Accuracy:       {self.metrics['prediction_accuracy']:8.1f}%
  Max Consecutive Losses:    {self.metrics['max_consecutive_losses']:8.0f}

PORTFOLIO METRICS:
  Final Portfolio Value:     ${self.metrics['final_value']:,.2f}
  Initial Capital:           ${self.config['initial_capital']:,.2f}
        """
        
        return summary.strip()

def run_backtest_analysis(predictions_df, config=None):
    """
    Convenience function to run complete backtest analysis
    
    Parameters:
    predictions_df (pd.DataFrame): Predictions with actual returns
    config (dict): Backtest configuration
    
    Returns:
    dict: Complete backtest results and analysis
    """
    print("Starting comprehensive backtest analysis...")
    
    # Initialize and run backtest
    engine = BacktestEngine(config)
    results = engine.run(predictions_df)
    
    # Get additional analysis
    trade_analysis = engine.get_trade_analysis()
    performance_summary = engine.get_performance_summary()
    
    print("\n" + performance_summary)
    
    if trade_analysis and trade_analysis['total'] > 0:
        print(f"\nTRADE ANALYSIS:")
        print(f"  Long trades:     {trade_analysis['long']['count']:4d} (Win rate: {trade_analysis['long']['win_rate']:5.1f}%)")
        print(f"  Short trades:    {trade_analysis['short']['count']:4d} (Win rate: {trade_analysis['short']['win_rate']:5.1f}%)")
        print(f"  Avg holding:     {trade_analysis['avg_holding_period']:5.1f} days")
        
        if trade_analysis['best_trade']:
            print(f"  Best trade:      {trade_analysis['best_trade']['direction']} ${trade_analysis['best_trade']['pnl']:,.0f} ({trade_analysis['best_trade']['return']:+.1f}%)")
        if trade_analysis['worst_trade']:
            print(f"  Worst trade:     {trade_analysis['worst_trade']['direction']} ${trade_analysis['worst_trade']['pnl']:,.0f} ({trade_analysis['worst_trade']['return']:+.1f}%)")
    
    return {
        'results': results,
        'trade_analysis': trade_analysis,
        'performance_summary': performance_summary,
        'engine': engine
    }

if __name__ == "__main__":
    # Example usage
    from ..models.ml_models import EnsembleAlphaModel
    from ..utils.alpha_factors import calculate_alpha_factors
    from ..utils.data_generator import generate_complete_dataset
    
    print("Running complete backtest example...")
    
    # Generate data and train model
    df = generate_complete_dataset()
    factors_df = calculate_alpha_factors(df)
    
    model = EnsembleAlphaModel(random_state=42)
    model_results = model.fit(factors_df)
    
    if model_results:
        predictions_df = model_results['predictions']
        
        # Configure backtest
        backtest_config = {
            'initial_capital': 100000,
            'transaction_cost': 0.001,
            'signal_threshold': 0.005,
            'max_position_size': 1.0,
            'risk_free_rate': 0.02
        }
        
        # Run backtest
        backtest_results = run_backtest_analysis(predictions_df, backtest_config)
        
        # Export results
        backtest_results['engine'].export_to_csv('backtest_results.csv')
        
        # Save trade details if any
        if not backtest_results['results']['trades'].empty:
            backtest_results['results']['trades'].to_csv('trade_details.csv', index=False)
            print("Trade details saved to trade_details.csv")
    
    else:
        print("Model training failed - cannot run backtest")