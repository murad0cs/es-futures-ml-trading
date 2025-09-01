import pandas as pd
import numpy as np
from typing import Tuple, List

# Try to import talib, use fallback implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_volatility_features(df)
        df = self._add_technical_indicators(df)
        df = self._add_microstructure_features(df)
        df = self._add_time_features(df)
        df = self._add_market_regime_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in [5, 10, 20, 60]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']
        
        df['pullback_depth'] = (df['high'].rolling(20).max() - df['close']) / df['high'].rolling(20).max()
        df['pullback_speed'] = df['returns'].rolling(5).mean() / df['returns'].rolling(5).std()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_surge'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['price_to_vwap'] = df['close'] / df['vwap'] - 1
        
        df['dollar_volume'] = df['close'] * df['volume']
        df['dollar_volume_ratio'] = df['dollar_volume'] / df['dollar_volume'].rolling(20).mean()
        
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_momentum'] = df['obv'] - df['obv'].shift(20)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if TALIB_AVAILABLE:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.config.atr_period)
        else:
            # Fallback ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(self.config.atr_period).mean()
        
        for period in [10, 20, 60]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Calculate volatility ratios after all volatilities are computed
        for period in [10, 60]:
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['volatility_20']
        
        df['parkinson_volatility'] = np.sqrt(
            np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        ).rolling(20).mean()
        
        df['garman_klass_volatility'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low']) ** 2 - 
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        ).rolling(20).mean()
        
        df['volatility_regime'] = pd.qcut(df['volatility_20'].rolling(60).mean(), 
                                         q=3, labels=['low', 'medium', 'high'])
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI calculation
        if TALIB_AVAILABLE:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD calculation
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
        else:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        df['macd_divergence'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if TALIB_AVAILABLE:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
        else:
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        df['bollinger_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bollinger_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        if TALIB_AVAILABLE:
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        else:
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ADX (simplified)
        if TALIB_AVAILABLE:
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # Simple trend strength proxy
            df['adx'] = abs(df['close'].rolling(14).mean() - df['close']) / df['atr'] * 10
            df['adx'] = df['adx'].rolling(14).mean()
        
        df['trend_strength'] = df['adx'] / 100
        
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'distance_to_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['tick_direction'] = np.sign(df['close'].diff())
        df['tick_distribution'] = df['tick_direction'].rolling(20).sum() / 20
        
        df['order_flow_imbalance'] = (df['volume'] * df['tick_direction']).rolling(10).sum()
        df['order_flow_ratio'] = df['order_flow_imbalance'] / df['volume'].rolling(10).sum()
        
        df['liquidity_score'] = 1 / (df['high'] - df['low'])
        df['liquidity_ratio'] = df['liquidity_score'] / df['liquidity_score'].rolling(20).mean()
        
        df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        df['time_of_day'] = df['hour'] + df['minute'] / 60
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_first_hour'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(int)
        df['is_last_hour'] = (df['hour'] == 15).astype(int)
        
        df['sin_time'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
        df['cos_time'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['market_regime_trend'] = np.where(
            df['sma_20'] > df['sma_50'], 'uptrend',
            np.where(df['sma_20'] < df['sma_50'], 'downtrend', 'sideways')
        )
        
        df['momentum_regime'] = pd.qcut(df['returns_20'].rolling(60).mean(), 
                                       q=3, labels=['bearish', 'neutral', 'bullish'])
        
        df['volatility_percentile'] = df['volatility_20'].rolling(252).rank(pct=True)
        
        support_window = 20
        resistance_window = 20
        df['support_level'] = df['low'].rolling(support_window).min()
        df['resistance_level'] = df['high'].rolling(resistance_window).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, 
                            target_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        if target_type == 'stop_loss':
            return self._prepare_stop_loss_data(df)
        elif target_type == 'entry_confidence':
            return self._prepare_entry_confidence_data(df)
        elif target_type == 'position_size':
            return self._prepare_position_size_data(df)
        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    def _prepare_stop_loss_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        
        future_low = df['low'].rolling(self.config.prediction_horizon).min().shift(-self.config.prediction_horizon)
        optimal_stop = future_low - (df['atr'] * 0.5)
        
        feature_cols = self.config.stop_loss_features
        X = df[feature_cols].dropna()
        y = optimal_stop.loc[X.index]
        
        return X, y
    
    def _prepare_entry_confidence_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        
        future_return = df['close'].shift(-self.config.prediction_horizon) / df['close'] - 1
        trade_success = (future_return > df['atr'] * 2 / df['close']).astype(int)
        
        feature_cols = self.config.entry_confidence_features
        X = df[feature_cols].dropna()
        y = trade_success.loc[X.index]
        
        return X, y
    
    def _prepare_position_size_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        
        volatility_normalized = df['volatility_20'] / df['volatility_20'].rolling(252).mean()
        optimal_position_size = 1 / volatility_normalized
        optimal_position_size = optimal_position_size.clip(0.5, 2.0)
        
        feature_cols = self.config.stop_loss_features + ['predicted_stop_loss', 'entry_confidence']
        X = df[feature_cols].dropna()
        y = optimal_position_size.loc[X.index]
        
        return X, y