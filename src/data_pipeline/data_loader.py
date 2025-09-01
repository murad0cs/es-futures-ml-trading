import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_data(self, symbol: str = 'ES=F', 
                 start_date: str = None, 
                 end_date: str = None,
                 interval: str = '1m') -> pd.DataFrame:
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print(f"No data retrieved for {symbol}. Generating synthetic data...")
                df = self.generate_synthetic_data(start_date, end_date)
            
            df.columns = df.columns.str.lower()
            
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}. Generating synthetic data...")
            return self.generate_synthetic_data(start_date, end_date)
    
    def generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start, end=end, freq='1min')
        
        date_range = date_range[(date_range.hour >= 9) & (date_range.hour < 16)]
        date_range = date_range[date_range.dayofweek < 5]
        
        n_periods = len(date_range)
        
        np.random.seed(self.config.random_seed)
        
        initial_price = 4500
        returns = np.random.normal(0.0001, 0.002, n_periods)
        
        volatility_regimes = np.random.choice([0.001, 0.002, 0.003], n_periods, p=[0.3, 0.5, 0.2])
        returns = returns * volatility_regimes
        
        trend = np.sin(np.linspace(0, 4*np.pi, n_periods)) * 0.0002
        returns = returns + trend
        
        prices = initial_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=date_range)
        df['close'] = prices
        
        daily_volatility = np.random.uniform(0.002, 0.01, n_periods)
        df['high'] = df['close'] * (1 + daily_volatility)
        df['low'] = df['close'] * (1 - daily_volatility)
        df['open'] = df['close'].shift(1).fillna(initial_price)
        
        df['volume'] = np.random.gamma(2, 50000, n_periods).astype(int)
        
        volume_spikes = np.random.choice([False, True], n_periods, p=[0.95, 0.05])
        df.loc[volume_spikes, 'volume'] *= np.random.uniform(2, 5)
        
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        try:
            vix = yf.Ticker('^VIX')
            vix_data = vix.history(start=df.index[0], end=df.index[-1], interval='1d')
            if not vix_data.empty:
                vix_data = vix_data['Close'].resample('1min').ffill()
                df['vix_level'] = df.index.map(lambda x: vix_data.get(x, 20))
            else:
                df['vix_level'] = 20 + np.random.normal(0, 2, len(df))
        except:
            df['vix_level'] = 20 + np.random.normal(0, 2, len(df))
        
        df['vix_regime'] = pd.cut(df['vix_level'], 
                                  bins=[0, 15, 25, 100], 
                                  labels=['low_vol', 'normal_vol', 'high_vol'])
        
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                               train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Train set: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"Test set: {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
        
        return train_df, test_df
    
    def generate_trade_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['sma_fast'] = df['close'].rolling(10).mean()
        df['sma_slow'] = df['close'].rolling(30).mean()
        
        df['signal'] = 0
        df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
        df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1
        
        df['trade_signal'] = df['signal'].diff()
        
        long_signals = df[df['trade_signal'] == 2].copy()
        long_signals['direction'] = 'long'
        
        signals = long_signals[['direction']].copy()
        signals['confidence_score'] = 1.0
        
        return signals