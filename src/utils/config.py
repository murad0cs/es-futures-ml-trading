import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TradingConfig:
    symbol: str = "ES"
    timeframe: str = "1min"
    
    initial_capital: float = 100000
    risk_per_trade: float = 0.01
    max_position_size: int = 10
    
    atr_period: int = 14
    atr_multiplier: float = 2.0
    
    lookback_periods: int = 120
    prediction_horizon: int = 30
    
    train_test_split: float = 0.8
    validation_split: float = 0.2
    
    random_seed: int = 42
    
    target_sharpe: float = 1.0
    target_calmar: float = 2.0
    target_profit_factor: float = 1.7
    target_win_rate: float = 0.55
    max_drawdown_threshold: float = 0.10
    target_win_loss_ratio: float = 2.0
    
    # Feature lists
    stop_loss_features: list = None
    entry_confidence_features: list = None
    
    def __post_init__(self):
        if self.stop_loss_features is None:
            self.stop_loss_features = [
                'atr', 'volatility_20', 'volatility_60',
                'volume_ratio', 'price_momentum_20',
                'rsi', 'macd', 'bollinger_width',
                'support_distance', 'resistance_distance',
                'vix_level', 'time_of_day', 'day_of_week'
            ]
        
        if self.entry_confidence_features is None:
            self.entry_confidence_features = [
                'pullback_depth', 'pullback_speed',
                'volume_surge', 'trend_strength',
                'macd_divergence', 'volatility_20',
                'volatility_regime', 'liquidity_score',
                'order_flow_imbalance', 'tick_distribution',
                'support_distance', 'resistance_distance'
            ]

@dataclass
class ModelConfig:
    xgb_params: Dict[str, Any] = None
    lgb_params: Dict[str, Any] = None
    nn_architecture: Dict[str, Any] = None
    validation_split: float = 0.2
    random_seed: int = 42
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        
        if self.lgb_params is None:
            self.lgb_params = {
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'regression',
                'random_state': 42
            }
        
        if self.nn_architecture is None:
            self.nn_architecture = {
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'activation': 'relu',
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }