import numpy as np
import pandas as pd
from typing import Dict, Any

class AdaptivePositionSizer:
    def __init__(self, config):
        self.config = config
        self.risk_budget = config.risk_per_trade
        self.max_position_size = config.max_position_size
        
    def calculate_position_size(self, 
                               entry_price: float,
                               stop_loss_price: float,
                               confidence_score: float,
                               account_balance: float,
                               volatility_regime: str = 'medium') -> Dict[str, Any]:
        
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {
                'position_size': 0,
                'dollar_risk': 0,
                'position_value': 0,
                'risk_reward_ratio': 0
            }
        
        dollar_risk = account_balance * self.risk_budget
        
        volatility_adjustment = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8
        }.get(volatility_regime, 1.0)
        
        confidence_adjustment = 0.5 + (confidence_score * 0.5)
        
        adjusted_dollar_risk = dollar_risk * volatility_adjustment * confidence_adjustment
        
        base_position_size = adjusted_dollar_risk / risk_per_share
        
        kelly_fraction = self._calculate_kelly_fraction(confidence_score)
        kelly_position_size = (account_balance * kelly_fraction) / entry_price
        
        position_size = min(base_position_size, kelly_position_size)
        
        position_size = min(position_size, self.max_position_size)
        position_size = max(1, int(position_size))
        
        actual_dollar_risk = position_size * risk_per_share
        position_value = position_size * entry_price
        
        target_price = entry_price + (2 * (entry_price - stop_loss_price))
        risk_reward_ratio = 2.0
        
        return {
            'position_size': position_size,
            'dollar_risk': actual_dollar_risk,
            'position_value': position_value,
            'risk_reward_ratio': risk_reward_ratio,
            'target_price': target_price,
            'volatility_adjustment': volatility_adjustment,
            'confidence_adjustment': confidence_adjustment,
            'kelly_fraction': kelly_fraction
        }
    
    def _calculate_kelly_fraction(self, win_probability: float, 
                                 avg_win_loss_ratio: float = 2.0) -> float:
        p = win_probability
        b = avg_win_loss_ratio
        
        if p <= 0 or p >= 1:
            return 0
        
        kelly = (p * b - (1 - p)) / b
        
        kelly = max(0, kelly)
        kelly = min(0.25, kelly)
        
        return kelly * 0.5
    
    def calculate_portfolio_heat(self, open_positions: pd.DataFrame, 
                                account_balance: float) -> Dict[str, float]:
        if len(open_positions) == 0:
            return {
                'total_heat': 0,
                'heat_percentage': 0,
                'positions_count': 0,
                'max_allowed_positions': self._calculate_max_positions(0)
            }
        
        total_risk = open_positions['dollar_risk'].sum()
        heat_percentage = (total_risk / account_balance) * 100
        
        return {
            'total_heat': total_risk,
            'heat_percentage': heat_percentage,
            'positions_count': len(open_positions),
            'max_allowed_positions': self._calculate_max_positions(heat_percentage)
        }
    
    def _calculate_max_positions(self, current_heat: float) -> int:
        max_heat = 6.0
        remaining_heat = max_heat - current_heat
        
        if remaining_heat <= 0:
            return 0
        
        positions_allowed = int(remaining_heat / (self.risk_budget * 100))
        return min(positions_allowed, 5)
    
    def adjust_for_correlation(self, position_sizes: Dict[str, float], 
                              correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        adjusted_sizes = {}
        
        for symbol in position_sizes:
            if symbol not in correlation_matrix.columns:
                adjusted_sizes[symbol] = position_sizes[symbol]
                continue
            
            correlations = correlation_matrix[symbol]
            existing_positions = [s for s in adjusted_sizes.keys() if s in correlations.index]
            
            if not existing_positions:
                adjusted_sizes[symbol] = position_sizes[symbol]
            else:
                avg_correlation = correlations[existing_positions].mean()
                
                correlation_adjustment = 1.0 - (abs(avg_correlation) * 0.5)
                correlation_adjustment = max(0.3, correlation_adjustment)
                
                adjusted_sizes[symbol] = position_sizes[symbol] * correlation_adjustment
        
        return adjusted_sizes
    
    def calculate_dynamic_stops(self, df: pd.DataFrame, 
                               predicted_volatility: pd.Series) -> pd.DataFrame:
        df = df.copy()
        
        df['dynamic_atr_multiplier'] = 2.0 - (predicted_volatility * 0.5)
        df['dynamic_atr_multiplier'] = df['dynamic_atr_multiplier'].clip(1.5, 2.5)
        
        df['dynamic_stop_distance'] = df['atr'] * df['dynamic_atr_multiplier']
        
        df['long_stop'] = df['close'] - df['dynamic_stop_distance']
        df['short_stop'] = df['close'] + df['dynamic_stop_distance']
        
        df['long_stop'] = df['long_stop'].rolling(3).mean()
        df['short_stop'] = df['short_stop'].rolling(3).mean()
        
        return df[['long_stop', 'short_stop', 'dynamic_stop_distance', 'dynamic_atr_multiplier']]