import pandas as pd
import numpy as np
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.entry_confidence_model import EntryConfidenceModel
from src.utils.config import TradingConfig, ModelConfig
from src.backtesting.backtest_engine import BacktestEngine

config = TradingConfig()
model_config = ModelConfig()
data_loader = DataLoader(config)
feature_engineer = FeatureEngineer(config)
backtest_engine = BacktestEngine(config)

# Load and prepare data
print("Loading data...")
df = data_loader.load_data(
    symbol='ES=F',
    start_date='2024-10-01',
    end_date='2024-12-31',
    interval='1h'
)

# Engineer features
df = feature_engineer.create_features(df)
df = data_loader.add_external_features(df)

# Split data
train_df, test_df = data_loader.prepare_train_test_split(df)

# Generate signals
train_signals = data_loader.generate_trade_signals(train_df)
test_signals = data_loader.generate_trade_signals(test_df)

print(f"\nTest signals generated: {len(test_signals)}")
if len(test_signals) > 0:
    print("Signal timestamps:")
    for idx in test_signals.index:
        print(f"  - {idx}")

# Check what the ML model would do
print("\n" + "="*50)
print("CHECKING ML CONFIDENCE FILTERING")
print("="*50)

# Prepare features for confidence model
X_confidence = test_df[config.entry_confidence_features].copy()
for col in X_confidence.columns:
    if X_confidence[col].dtype.name == 'category':
        X_confidence[col] = X_confidence[col].cat.codes
    elif X_confidence[col].dtype == 'object':
        X_confidence[col] = pd.Categorical(X_confidence[col]).codes
X_confidence = X_confidence.fillna(0)
X_confidence = X_confidence.replace([np.inf, -np.inf], 0)

print(f"\nConfidence features shape: {X_confidence.shape}")
print(f"Test signals indices: {test_signals.index.tolist()}")

# Create a simple mock confidence model that returns high confidence
class MockConfidenceModel:
    def predict_proba(self, X):
        # Return high confidence (0.8) for all samples
        return np.ones(len(X)) * 0.8

mock_model = MockConfidenceModel()
confidence_scores = mock_model.predict_proba(X_confidence)

# Convert to series and filter
confidence_series = pd.Series(confidence_scores, index=test_df.index)
signal_confidence_scores = confidence_series.loc[test_signals.index]

print(f"\nConfidence scores for signals:")
for idx, score in zip(test_signals.index, signal_confidence_scores):
    print(f"  {idx}: {score:.3f}")

filtered_signals = test_signals[signal_confidence_scores >= 0.5]
print(f"\nSignals passing confidence filter (>= 0.5): {len(filtered_signals)}")

# Test baseline backtest
print("\n" + "="*50)
print("TESTING BASELINE BACKTEST")
print("="*50)

baseline_result = backtest_engine.run_backtest(
    df=test_df,
    signals=test_signals
)

print(f"Baseline trades executed: {len(baseline_result['trades'])}")
print(f"Baseline total return: {baseline_result['total_return']*100:.2f}%")
print(f"Baseline final capital: ${baseline_result['final_capital']:,.2f}")

if len(baseline_result['trades']) > 0:
    print("\nFirst 3 trades:")
    print(baseline_result['trades'].head(3))
else:
    print("\nNo trades were executed in baseline backtest!")
    
    # Debug the backtest logic
    print("\nDebugging backtest logic...")
    print(f"Initial capital: ${config.initial_capital:,.2f}")
    print(f"Commission: {backtest_engine.commission:.4f}")
    print(f"Slippage: {backtest_engine.slippage:.4f}")
    
    # Check if signals align with DataFrame indices
    print(f"\nSignal indices in DataFrame?: {all(idx in test_df.index for idx in test_signals.index)}")
    
    # Check ATR for stop losses
    if 'atr' in test_df.columns:
        print(f"ATR range: {test_df['atr'].min():.2f} - {test_df['atr'].max():.2f}")
        print(f"ATR multiplier: {config.atr_multiplier}")
    else:
        print("No ATR column found in test_df!")