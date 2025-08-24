"""Performance analytics and visualization tools"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class PerformanceAnalytics:
    """Performance analytics for backtesting results"""
    
    def __init__(self, results: pd.DataFrame, benchmark: Optional[pd.Series] = None):
        """
        Initialize performance analytics
        
        Args:
            results: DataFrame with backtest results including 'returns' column
            benchmark: Optional benchmark returns series
        """
        self.results = results.copy()
        self.benchmark = benchmark
        
        if 'returns' not in self.results.columns:
            raise ValueError("Results must contain 'returns' column")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        returns = self.results['returns'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic returns metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Risk-adjusted returns
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
        
        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = returns.quantile(0.05)
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'VaR (95%)': var_95,
            'Downside Deviation': downside_deviation
        }
        
        # Benchmark comparison if available
        if self.benchmark is not None:
            aligned_benchmark = self.benchmark.reindex(returns.index).dropna()
            if len(aligned_benchmark) > 0:
                benchmark_return = (1 + aligned_benchmark).prod() - 1
                benchmark_vol = aligned_benchmark.std() * np.sqrt(252)
                
                # Beta calculation
                covariance = returns.cov(aligned_benchmark)
                beta = covariance / (aligned_benchmark.var()) if aligned_benchmark.var() != 0 else 0
                
                # Alpha calculation
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                alpha = annualized_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
                
                # Information ratio
                excess_returns = returns - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error != 0 else 0
                
                metrics.update({
                    'Alpha': alpha,
                    'Beta': beta,
                    'Information Ratio': information_ratio,
                    'Tracking Error': tracking_error,
                    'Benchmark Return': benchmark_return,
                    'Benchmark Volatility': benchmark_vol
                })
        
        return metrics
    
    def create_performance_summary(self) -> pd.DataFrame:
        """Create a formatted performance summary"""
        metrics = self.calculate_metrics()
        
        # Format metrics nicely
        formatted_metrics = {}
        for key, value in metrics.items():
            if 'Return' in key or 'Drawdown' in key or 'VaR' in key or 'Alpha' in key:
                formatted_metrics[key] = f"{value:.2%}"
            elif 'Ratio' in key or 'Beta' in key or 'Factor' in key:
                formatted_metrics[key] = f"{value:.2f}"
            elif 'Rate' in key:
                formatted_metrics[key] = f"{value:.2%}"
            else:
                formatted_metrics[key] = f"{value:.4f}"
        
        return pd.DataFrame.from_dict(formatted_metrics, orient='index', columns=['Value'])
    
    def plot_performance(self, figsize: tuple = (15, 10)) -> plt.Figure:
        """Create comprehensive performance plots"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Backtest Performance Analysis', fontsize=16)
        
        # Cumulative returns
        cumulative_returns = (1 + self.results['returns'].fillna(0)).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, label='Strategy', linewidth=2)
        
        if self.benchmark is not None:
            benchmark_aligned = self.benchmark.reindex(cumulative_returns.index).fillna(0)
            benchmark_cumulative = (1 + benchmark_aligned).cumprod()
            axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                          label='Benchmark', linewidth=2, alpha=0.7)
            axes[0, 0].legend()
        
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = (1 + self.results['returns'].fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        monthly_returns = self.results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_matrix = monthly_returns.groupby([monthly_returns.index.year, 
                                                        monthly_returns.index.month]).first().unstack()
        
        if not monthly_returns_matrix.empty:
            im = axes[1, 0].imshow(monthly_returns_matrix.values, cmap='RdYlGn', aspect='auto')
            axes[1, 0].set_title('Monthly Returns Heatmap')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Year')
            
            # Set ticks
            axes[1, 0].set_xticks(range(12))
            axes[1, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            axes[1, 0].set_yticks(range(len(monthly_returns_matrix.index)))
            axes[1, 0].set_yticklabels(monthly_returns_matrix.index)
            
            plt.colorbar(im, ax=axes[1, 0])
        
        # Rolling Sharpe ratio
        rolling_returns = self.results['returns'].rolling(window=63).mean() * 252
        rolling_vol = self.results['returns'].rolling(window=63).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Rolling Sharpe Ratio (3 months)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_performance(self) -> go.Figure:
        """Create interactive performance dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Portfolio Value', 
                          'Drawdown', 'Rolling Sharpe Ratio',
                          'Monthly Returns Distribution', 'Leverage'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cumulative returns
        cumulative_returns = (1 + self.results['returns'].fillna(0)).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                      name='Strategy', line=dict(width=2)),
            row=1, col=1
        )
        
        if self.benchmark is not None:
            benchmark_aligned = self.benchmark.reindex(cumulative_returns.index).fillna(0)
            benchmark_cumulative = (1 + benchmark_aligned).cumprod()
            fig.add_trace(
                go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative.values,
                          name='Benchmark', line=dict(width=2)),
                row=1, col=1
            )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=self.results.index, y=self.results['portfolio_value'],
                      name='Portfolio Value', line=dict(width=2, color='green')),
            row=1, col=2
        )
        
        # Drawdown
        cumulative = (1 + self.results['returns'].fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      fill='tonexty', name='Drawdown', 
                      line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
            row=2, col=1
        )
        
        # Rolling Sharpe
        rolling_returns = self.results['returns'].rolling(window=63).mean() * 252
        rolling_vol = self.results['returns'].rolling(window=63).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Rolling Sharpe', line=dict(width=2, color='purple')),
            row=2, col=2
        )
        
        # Monthly returns distribution
        monthly_returns = self.results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        fig.add_trace(
            go.Histogram(x=monthly_returns.values, nbinsx=20, name='Monthly Returns'),
            row=3, col=1
        )
        
        # Leverage
        if 'leverage' in self.results.columns:
            fig.add_trace(
                go.Scatter(x=self.results.index, y=self.results['leverage'],
                          name='Leverage', line=dict(width=2, color='orange')),
                row=3, col=2
            )
        
        fig.update_layout(height=900, showlegend=True, title_text="Interactive Performance Dashboard")
        
        return fig
    
    def generate_tearsheet(self) -> Dict[str, Any]:
        """Generate a comprehensive performance tearsheet"""
        metrics = self.calculate_metrics()
        summary_df = self.create_performance_summary()
        
        tearsheet = {
            'metrics': metrics,
            'summary_table': summary_df,
            'performance_plot': self.plot_performance(),
            'interactive_plot': self.plot_interactive_performance()
        }
        
        return tearsheet