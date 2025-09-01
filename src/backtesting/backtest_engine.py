import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: int
    direction: str
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    confidence_score: float = 1.0
    
class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.initial_capital
        self.commission = 0.0005
        self.slippage = 0.0001
        
    def run_backtest(self, df: pd.DataFrame, 
                    signals: pd.DataFrame,
                    stop_losses: pd.Series = None,
                    confidence_scores: pd.Series = None,
                    position_sizes: pd.Series = None) -> Dict[str, Any]:
        
        df = df.copy()
        signals = signals.copy()
        
        if stop_losses is not None:
            df['stop_loss'] = stop_losses
        else:
            df['stop_loss'] = df['close'] - (df['atr'] * self.config.atr_multiplier)
        
        if confidence_scores is not None:
            df['confidence_score'] = confidence_scores
        else:
            df['confidence_score'] = 1.0
        
        if position_sizes is not None:
            df['position_size'] = position_sizes
        else:
            df['position_size'] = 1
        
        trades = []
        equity_curve = [self.initial_capital]
        current_position = None
        current_capital = self.initial_capital
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            if current_position is None and i in signals.index:
                signal = signals.loc[i]
                
                if signal.get('confidence_score', 1.0) < 0.5:
                    continue
                
                entry_price = current_row['close'] * (1 + self.slippage)
                stop_loss = current_row['stop_loss']
                take_profit = entry_price + (2 * (entry_price - stop_loss))
                position_size = min(
                    current_row['position_size'],
                    int(current_capital * 0.1 / entry_price)
                )
                
                if position_size > 0:
                    current_position = {
                        'entry_time': current_row.name,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'direction': signal.get('direction', 'long'),
                        'confidence_score': signal.get('confidence_score', 1.0)
                    }
            
            elif current_position is not None:
                exit_signal = False
                exit_price = None
                exit_reason = None
                
                if current_position['direction'] == 'long':
                    if current_row['low'] <= current_position['stop_loss']:
                        exit_signal = True
                        exit_price = current_position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_row['high'] >= current_position['take_profit']:
                        exit_signal = True
                        exit_price = current_position['take_profit']
                        exit_reason = 'take_profit'
                    elif i == len(df) - 1:
                        exit_signal = True
                        exit_price = current_row['close']
                        exit_reason = 'end_of_data'
                
                if exit_signal:
                    exit_price = exit_price * (1 - self.slippage)
                    
                    if current_position['direction'] == 'long':
                        pnl = (exit_price - current_position['entry_price']) * current_position['position_size']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) * current_position['position_size']
                    
                    commission_cost = (current_position['entry_price'] + exit_price) * \
                                    current_position['position_size'] * self.commission
                    pnl -= commission_cost
                    
                    pnl_pct = pnl / (current_position['entry_price'] * current_position['position_size'])
                    
                    trade = Trade(
                        entry_time=current_position['entry_time'],
                        exit_time=current_row.name,
                        entry_price=current_position['entry_price'],
                        exit_price=exit_price,
                        position_size=current_position['position_size'],
                        direction=current_position['direction'],
                        stop_loss=current_position['stop_loss'],
                        take_profit=current_position['take_profit'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        confidence_score=current_position['confidence_score']
                    )
                    
                    trades.append(trade)
                    current_capital += pnl
                    current_position = None
            
            equity_curve.append(current_capital)
        
        trades_df = pd.DataFrame([vars(t) for t in trades])
        if len(trades_df) > 0:
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
        
        equity_series = pd.Series(equity_curve[:len(df)], index=df.index)
        returns = equity_series.pct_change().dropna()
        
        return {
            'trades': trades_df,
            'equity_curve': equity_series,
            'returns': returns,
            'final_capital': current_capital,
            'total_return': (current_capital - self.initial_capital) / self.initial_capital
        }
    
    def run_ml_enhanced_backtest(self, df: pd.DataFrame,
                                signals: pd.DataFrame,
                                stop_loss_model,
                                entry_confidence_model,
                                position_sizer) -> Dict[str, Any]:
        
        df = df.copy()
        
        X_stop_loss = df[self.config.stop_loss_features].fillna(0)
        dynamic_stops = stop_loss_model.predict(X_stop_loss)
        
        X_confidence = df[self.config.entry_confidence_features].fillna(0)
        confidence_scores = entry_confidence_model.predict_proba(X_confidence)
        
        filtered_signals = signals[confidence_scores >= 0.5]
        
        position_sizes = []
        for i in range(len(df)):
            if i in filtered_signals.index:
                pos_size_result = position_sizer.calculate_position_size(
                    entry_price=df.iloc[i]['close'],
                    stop_loss_price=dynamic_stops[i],
                    confidence_score=confidence_scores[i],
                    account_balance=self.initial_capital,
                    volatility_regime=df.iloc[i].get('volatility_regime', 'medium')
                )
                position_sizes.append(pos_size_result['position_size'])
            else:
                position_sizes.append(0)
        
        position_sizes = pd.Series(position_sizes, index=df.index)
        
        return self.run_backtest(
            df=df,
            signals=filtered_signals,
            stop_losses=pd.Series(dynamic_stops, index=df.index),
            confidence_scores=pd.Series(confidence_scores, index=df.index),
            position_sizes=position_sizes
        )
    
    def run_walk_forward_analysis(self, df: pd.DataFrame,
                                 signals: pd.DataFrame,
                                 stop_loss_model,
                                 entry_confidence_model,
                                 position_sizer,
                                 window_size: int = 5000,
                                 step_size: int = 1000) -> Dict[str, Any]:
        
        results = []
        equity_curve = [self.initial_capital]
        all_trades = []
        
        for start_idx in range(0, len(df) - window_size, step_size):
            end_idx = min(start_idx + window_size, len(df))
            
            window_df = df.iloc[start_idx:end_idx]
            window_signals = signals.loc[signals.index.intersection(window_df.index)]
            
            if len(window_signals) == 0:
                continue
            
            window_result = self.run_ml_enhanced_backtest(
                df=window_df,
                signals=window_signals,
                stop_loss_model=stop_loss_model,
                entry_confidence_model=entry_confidence_model,
                position_sizer=position_sizer
            )
            
            results.append({
                'start': window_df.index[0],
                'end': window_df.index[-1],
                'total_return': window_result['total_return'],
                'n_trades': len(window_result['trades'])
            })
            
            if len(window_result['trades']) > 0:
                all_trades.append(window_result['trades'])
            
            if len(equity_curve) > 0:
                last_equity = equity_curve[-1]
                window_equity = window_result['equity_curve'] * (last_equity / self.initial_capital)
                equity_curve.extend(window_equity.tolist()[1:])
        
        return {
            'window_results': pd.DataFrame(results),
            'all_trades': pd.concat(all_trades) if all_trades else pd.DataFrame(),
            'equity_curve': pd.Series(equity_curve),
            'final_return': (equity_curve[-1] - self.initial_capital) / self.initial_capital if equity_curve else 0
        }
    
    def calculate_trade_analysis(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        if len(trades_df) == 0:
            return {}
        
        analysis = {
            'by_exit_reason': trades_df.groupby('exit_reason').agg({
                'pnl': ['count', 'sum', 'mean'],
                'pnl_pct': 'mean'
            }),
            
            'by_confidence': pd.qcut(trades_df['confidence_score'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High']).value_counts(),
            
            'hourly_performance': trades_df.set_index('entry_time').groupby(lambda x: x.hour).agg({
                'pnl': 'mean',
                'pnl_pct': 'mean'
            }),
            
            'trade_duration_analysis': {
                'short_trades': trades_df[trades_df['duration'] < 30],
                'medium_trades': trades_df[(trades_df['duration'] >= 30) & (trades_df['duration'] < 120)],
                'long_trades': trades_df[trades_df['duration'] >= 120]
            }
        }
        
        return analysis