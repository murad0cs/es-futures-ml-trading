import pandas as pd
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.utils.config import TradingConfig

config = TradingConfig()
data_loader = DataLoader(config)
feature_engineer = FeatureEngineer(config)

# Load and prepare data
print("Loading data...")
df = data_loader.load_data(
    symbol='ES=F',
    start_date='2024-10-01',
    end_date='2024-12-31',
    interval='1h'
)
print(f"Loaded {len(df)} data points")

# Engineer features
print("\nEngineering features...")
df = feature_engineer.create_features(df)
df = data_loader.add_external_features(df)

# Split data
train_df, test_df = data_loader.prepare_train_test_split(df)
print(f"\nTrain set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")

# Generate signals
print("\nGenerating signals...")
train_signals = data_loader.generate_trade_signals(train_df)
test_signals = data_loader.generate_trade_signals(test_df)

print(f"\nTrain signals: {len(train_signals)}")
print(f"Test signals: {len(test_signals)}")

if len(test_signals) > 0:
    print("\nTest signal details:")
    print(test_signals.head())
    print(f"\nSignal indices: {test_signals.index.tolist()}")
else:
    print("\nNo test signals generated!")

# Check why so few signals
print("\nAnalyzing signal generation conditions...")
print(f"RSI oversold (<30) count: {(test_df['rsi'] < 30).sum()}")
print(f"RSI overbought (>70) count: {(test_df['rsi'] > 70).sum()}")
print(f"Bollinger band lower touches: {(test_df['close'] < test_df['bollinger_lower']).sum()}")
print(f"Bollinger band upper touches: {(test_df['close'] > test_df['bollinger_upper']).sum()}")
print(f"Volume surge (>1.5x avg) count: {(test_df['volume_ratio'] > 1.5).sum() if 'volume_ratio' in test_df.columns else 'N/A'}")