import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats

class PerformanceMetrics:
    def __init__(self):
        self.trading_days_per_year = 252
        self.minutes_per_day = 390
        
    def calculate_all_metrics(self, returns: pd.Series, 
                            trades: pd.DataFrame) -> Dict[str, Any]:
        metrics = {}
        
        metrics.update(self._calculate_return_metrics(returns))
        metrics.update(self._calculate_risk_metrics(returns))
        metrics.update(self._calculate_trade_metrics(trades))
        metrics.update(self._calculate_distribution_metrics(returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        total_return = (1 + returns).prod() - 1
        
        periods_per_year = self.trading_days_per_year * self.minutes_per_day
        n_periods = len(returns)
        years = n_periods / periods_per_year
        
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'cumulative_return': total_return * 100
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        periods_per_year = self.trading_days_per_year * self.minutes_per_day
        annual_vol = returns.std() * np.sqrt(periods_per_year)
        
        annual_return = self._calculate_return_metrics(returns)['annual_return'] / 100
        
        if annual_vol > 0:
            sharpe_ratio = annual_return / annual_vol
        else:
            sharpe_ratio = 0
        
        if max_drawdown != 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
            if downside_vol > 0:
                sortino_ratio = annual_return / downside_vol
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = float('inf')
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        drawdown_squared = drawdown ** 2
        ulcer_index = np.sqrt(drawdown_squared.mean()) * 100
        
        recovery_periods = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                recovery_periods.append(i - drawdown_start)
                in_drawdown = False
        
        avg_recovery_time = np.mean(recovery_periods) if recovery_periods else 0
        
        return {
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'annual_volatility': annual_vol * 100,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'ulcer_index': ulcer_index,
            'avg_recovery_time': avg_recovery_time
        }
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'win_loss_ratio': 0,
                'avg_trade_return': 0,
                'avg_trade_duration': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        avg_trade_return = trades['pnl_pct'].mean() if 'pnl_pct' in trades else 0
        avg_trade_duration = trades['duration'].mean() if 'duration' in trades else 0
        
        is_win = (trades['pnl'] > 0).astype(int)
        consecutive_wins = self._max_consecutive(is_win, 1)
        consecutive_losses = self._max_consecutive(is_win, 0)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'avg_trade_return': avg_trade_return * 100,
            'avg_trade_duration': avg_trade_duration,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            gain_to_pain_ratio = positive_returns.sum() / abs(negative_returns.sum())
        else:
            gain_to_pain_ratio = 0
        
        hit_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'gain_to_pain_ratio': gain_to_pain_ratio,
            'hit_rate': hit_rate * 100
        }
    
    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        groups = (series != value).cumsum()
        counts = series.groupby(groups).sum()
        return counts.max() if len(counts) > 0 else 0
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                 window: int = 252) -> pd.DataFrame:
        rolling_returns = returns.rolling(window)
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        rolling_metrics['rolling_return'] = rolling_returns.apply(
            lambda x: (1 + x).prod() - 1
        )
        
        rolling_metrics['rolling_volatility'] = rolling_returns.std() * np.sqrt(252)
        
        rolling_metrics['rolling_sharpe'] = (
            rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
        )
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_metrics['rolling_drawdown'] = (cumulative - rolling_max) / rolling_max
        
        return rolling_metrics
    
    def compare_strategies(self, baseline_metrics: Dict[str, float],
                          ml_metrics: Dict[str, float]) -> pd.DataFrame:
        comparison = pd.DataFrame({
            'Baseline': baseline_metrics,
            'ML Enhanced': ml_metrics
        })
        
        comparison['Improvement (%)'] = (
            (comparison['ML Enhanced'] - comparison['Baseline']) / 
            comparison['Baseline'].abs() * 100
        )
        
        comparison['Improvement (%)'] = comparison['Improvement (%)'].replace(
            [np.inf, -np.inf], np.nan
        )
        
        return comparison