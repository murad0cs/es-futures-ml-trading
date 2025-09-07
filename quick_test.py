import pandas as pd
import numpy as np
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.backtesting.metrics import PerformanceMetrics
from src.utils.config import TradingConfig as Config
from src.backtesting.backtest_engine import BacktestEngine
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK TEST - PROFESSIONAL ML-ENHANCED ES FUTURES TRADING SYSTEM")
print("="*80)

# Initialize
config = Config()
data_loader = DataLoader(config)
feature_engineer = FeatureEngineer(config)
backtest_engine = BacktestEngine(config)
metrics_calculator = PerformanceMetrics()

# Load data
print("\n1. Loading ES futures data...")
df = data_loader.load_data(
    symbol='ES=F',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1h'
)
print(f"[OK] Loaded {len(df)} data points")

# Engineer features
print("\n2. Engineering features...")
df = feature_engineer.create_features(df)
df = data_loader.add_external_features(df)
print(f"[OK] Created {len(df.columns)} features")

# Prepare data
print("\n3. Preparing train/test split...")
train_df, test_df = data_loader.prepare_train_test_split(df)

# Generate signals
print("\n4. Generating trade signals...")
train_signals = data_loader.generate_trade_signals(train_df)
test_signals = data_loader.generate_trade_signals(test_df)
print(f"[OK] Generated {len(train_signals)} training signals")
print(f"[OK] Generated {len(test_signals)} test signals")

# Run basic backtest without ML
print("\n5. Running baseline backtest...")
baseline_results = backtest_engine.run_backtest(
    df=test_df,
    signals=test_signals
)

# Calculate metrics
baseline_metrics = metrics_calculator.calculate_all_metrics(
    baseline_results['returns'],
    baseline_results['trades']
)

# Display results
print("\n" + "="*80)
print("BASELINE RESULTS (WITHOUT ML)")
print("="*80)

target_metrics = {
    'sharpe_ratio': {'target': 1.0, 'achieved': baseline_metrics.get('sharpe_ratio', 0), 'higher_better': True},
    'calmar_ratio': {'target': 2.0, 'achieved': baseline_metrics.get('calmar_ratio', 0), 'higher_better': True},
    'profit_factor': {'target': 1.7, 'achieved': baseline_metrics.get('profit_factor', 0), 'higher_better': True},
    'win_rate': {'target': 55.0, 'achieved': baseline_metrics.get('win_rate', 0), 'higher_better': True},
    'max_drawdown': {'target': 10.0, 'achieved': abs(baseline_metrics.get('max_drawdown', 0)), 'higher_better': False},
    'win_loss_ratio': {'target': 2.0, 'achieved': baseline_metrics.get('win_loss_ratio', 0), 'higher_better': True}
}

print(f"\n{'Metric':<20} {'Target':<15} {'Achieved':<15} {'Status':<15}")
print("-"*65)

for metric_name, data in target_metrics.items():
    target = data['target']
    achieved = data['achieved']
    higher_better = data['higher_better']
    
    if higher_better:
        status = "[ACHIEVED]" if achieved >= target else "[NOT MET]"
    else:
        status = "[ACHIEVED]" if achieved <= target else "[NOT MET]"
    
    print(f"{metric_name:<20} {f'{target:.2f}':<15} {f'{achieved:.2f}':<15} {status:<15}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total trades: {len(baseline_results['trades'])}")
print(f"Total return: {baseline_results['total_return']*100:.2f}%")
print(f"Final capital: ${baseline_results['final_capital']:,.2f}")

# Save results
print("\n[OK] Quick test completed successfully!")