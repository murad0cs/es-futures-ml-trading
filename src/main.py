import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Ensure proper path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.config import TradingConfig, ModelConfig
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.stop_loss_model import DynamicStopLossModel
from src.models.entry_confidence_model import EntryConfidenceModel
from src.models.position_sizing import AdaptivePositionSizer
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.metrics import PerformanceMetrics

class TradingSystemPipeline:
    def __init__(self):
        self.config = TradingConfig()
        self.model_config = ModelConfig()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        self.metrics_calculator = PerformanceMetrics()
        
    def run_complete_pipeline(self):
        print("="*80)
        print("ES FUTURES ML TRADING SYSTEM - COMPLETE PIPELINE")
        print("="*80)
        
        print("\n1. Loading and preparing data...")
        df = self.data_loader.load_data(
            symbol='ES=F',
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        print(f"Loaded {len(df)} data points")
        
        print("\n2. Engineering features...")
        df = self.feature_engineer.create_features(df)
        df = self.data_loader.add_external_features(df)
        print(f"Created {len(df.columns)} features")
        
        train_df, test_df = self.data_loader.prepare_train_test_split(df)
        
        print("\n3. Generating trade signals...")
        train_signals = self.data_loader.generate_trade_signals(train_df)
        test_signals = self.data_loader.generate_trade_signals(test_df)
        print(f"Generated {len(train_signals)} training signals and {len(test_signals)} test signals")
        
        print("\n" + "="*80)
        print("PHASE 1: DYNAMIC STOP-LOSS MODEL")
        print("="*80)
        
        stop_loss_model = DynamicStopLossModel(self.model_config)
        X_sl, y_sl = self.feature_engineer.prepare_training_data(train_df, 'stop_loss')
        
        print("Training stop-loss models...")
        sl_results = stop_loss_model.train(X_sl.fillna(0), y_sl.fillna(train_df['close'].mean() - train_df['atr'].mean() * 2))
        
        print("\nStop-Loss Model Performance:")
        for model_name, metrics in sl_results.items():
            print(f"  {model_name}:")
            print(f"    - RMSE: {metrics['val_rmse']:.4f}")
            print(f"    - MAE: {metrics['val_mae']:.4f}")
        
        print(f"\nBest model: {stop_loss_model.best_model_name}")
        
        print("\n" + "="*80)
        print("PHASE 2: ENTRY CONFIDENCE MODEL")
        print("="*80)
        
        entry_model = EntryConfidenceModel(self.model_config)
        X_ec, y_ec = self.feature_engineer.prepare_training_data(train_df, 'entry_confidence')
        
        print("Training entry confidence models...")
        ec_results = entry_model.train(X_ec.fillna(0), y_ec.fillna(0))
        
        print("\nEntry Confidence Model Performance:")
        for model_name, metrics in ec_results.items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics['val_accuracy']:.2%}")
            print(f"    - ROC-AUC: {metrics['val_roc_auc']:.4f}")
            print(f"    - Precision: {metrics['val_precision']:.2%}")
            print(f"    - Recall: {metrics['val_recall']:.2%}")
        
        print(f"\nBest model: {entry_model.best_model_name}")
        
        print("\n" + "="*80)
        print("PHASE 3: POSITION SIZING")
        print("="*80)
        
        position_sizer = AdaptivePositionSizer(self.config)
        print("Position sizing module initialized")
        
        print("\n" + "="*80)
        print("BACKTESTING: BASELINE vs ML-ENHANCED")
        print("="*80)
        
        print("\n4. Running baseline backtest...")
        baseline_result = self.backtest_engine.run_backtest(
            df=test_df,
            signals=test_signals
        )
        
        baseline_metrics = self.metrics_calculator.calculate_all_metrics(
            baseline_result['returns'],
            baseline_result['trades']
        )
        
        print("\n5. Running ML-enhanced backtest...")
        ml_result = self.backtest_engine.run_ml_enhanced_backtest(
            df=test_df,
            signals=test_signals,
            stop_loss_model=stop_loss_model,
            entry_confidence_model=entry_model,
            position_sizer=position_sizer
        )
        
        ml_metrics = self.metrics_calculator.calculate_all_metrics(
            ml_result['returns'],
            ml_result['trades']
        )
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        comparison = self.metrics_calculator.compare_strategies(
            baseline_metrics,
            ml_metrics
        )
        
        key_metrics = [
            'sharpe_ratio', 'calmar_ratio', 'profit_factor', 
            'win_rate', 'win_loss_ratio', 'max_drawdown',
            'total_return', 'kurtosis', 'skewness'
        ]
        
        print("\nKey Performance Metrics:")
        print("-" * 60)
        for metric in key_metrics:
            if metric in comparison.index:
                row = comparison.loc[metric]
                print(f"{metric:20s}: Baseline: {row['Baseline']:8.2f} | "
                      f"ML: {row['ML Enhanced']:8.2f} | "
                      f"Improvement: {row['Improvement (%)']:+7.1f}%")
        
        print("\n" + "="*80)
        print("TARGET METRICS ACHIEVEMENT")
        print("="*80)
        
        targets = {
            'sharpe_ratio': (1.0, ml_metrics.get('sharpe_ratio', 0)),
            'calmar_ratio': (2.0, ml_metrics.get('calmar_ratio', 0)),
            'profit_factor': (1.7, ml_metrics.get('profit_factor', 0)),
            'win_rate': (55, ml_metrics.get('win_rate', 0)),
            'max_drawdown': (10, abs(ml_metrics.get('max_drawdown', 100))),
            'win_loss_ratio': (2.0, ml_metrics.get('win_loss_ratio', 0))
        }
        
        for metric, (target, achieved) in targets.items():
            if metric == 'max_drawdown':
                status = "✓ ACHIEVED" if achieved <= target else "✗ NOT MET"
            else:
                status = "✓ ACHIEVED" if achieved >= target else "✗ NOT MET"
            
            print(f"{metric:20s}: Target: {target:6.2f} | Achieved: {achieved:6.2f} | {status}")
        
        print("\n" + "="*80)
        print("TRADE ANALYSIS")
        print("="*80)
        
        if len(ml_result['trades']) > 0:
            print(f"\nTotal Trades: {len(ml_result['trades'])}")
            print(f"Avg Trade Duration: {ml_result['trades']['duration'].mean():.1f} minutes")
            print(f"Trades Filtered by ML: {len(test_signals) - len(ml_result['trades'])}")
            
            print("\nTrade Outcomes:")
            for reason, count in ml_result['trades']['exit_reason'].value_counts().items():
                percentage = (count / len(ml_result['trades'])) * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")
        
        self.generate_report(baseline_result, ml_result, baseline_metrics, ml_metrics)
        
        return {
            'baseline_metrics': baseline_metrics,
            'ml_metrics': ml_metrics,
            'comparison': comparison,
            'models': {
                'stop_loss': stop_loss_model,
                'entry_confidence': entry_model,
                'position_sizer': position_sizer
            }
        }
    
    def generate_report(self, baseline_result, ml_result, baseline_metrics, ml_metrics):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].plot(baseline_result['equity_curve'], label='Baseline', alpha=0.7)
        axes[0, 0].plot(ml_result['equity_curve'], label='ML Enhanced', alpha=0.7)
        axes[0, 0].set_title('Equity Curves')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        baseline_dd = (baseline_result['equity_curve'] / baseline_result['equity_curve'].cummax() - 1) * 100
        ml_dd = (ml_result['equity_curve'] / ml_result['equity_curve'].cummax() - 1) * 100
        
        axes[0, 1].fill_between(baseline_dd.index, baseline_dd, 0, alpha=0.3, label='Baseline')
        axes[0, 1].fill_between(ml_dd.index, ml_dd, 0, alpha=0.3, label='ML Enhanced')
        axes[0, 1].set_title('Drawdown Comparison')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if len(ml_result['trades']) > 0:
            ml_result['trades']['pnl_pct'].hist(bins=30, alpha=0.7, ax=axes[0, 2])
            axes[0, 2].set_title('ML Trade Returns Distribution')
            axes[0, 2].set_xlabel('Return (%)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        metrics_comparison = pd.DataFrame({
            'Baseline': [baseline_metrics.get('sharpe_ratio', 0),
                        baseline_metrics.get('calmar_ratio', 0),
                        baseline_metrics.get('profit_factor', 0),
                        baseline_metrics.get('win_rate', 0) / 100],
            'ML Enhanced': [ml_metrics.get('sharpe_ratio', 0),
                           ml_metrics.get('calmar_ratio', 0),
                           ml_metrics.get('profit_factor', 0),
                           ml_metrics.get('win_rate', 0) / 100]
        }, index=['Sharpe', 'Calmar', 'Profit Factor', 'Win Rate'])
        
        metrics_comparison.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Key Metrics Comparison')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        baseline_cumret = (1 + baseline_result['returns']).cumprod()
        ml_cumret = (1 + ml_result['returns']).cumprod()
        
        baseline_rolling_vol = baseline_result['returns'].rolling(252).std() * np.sqrt(252) * 100
        ml_rolling_vol = ml_result['returns'].rolling(252).std() * np.sqrt(252) * 100
        
        axes[1, 1].plot(baseline_rolling_vol, label='Baseline', alpha=0.7)
        axes[1, 1].plot(ml_rolling_vol, label='ML Enhanced', alpha=0.7)
        axes[1, 1].set_title('Rolling Volatility (252 periods)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Volatility (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        win_loss_data = pd.DataFrame({
            'Avg Win': [baseline_metrics.get('avg_win', 0), ml_metrics.get('avg_win', 0)],
            'Avg Loss': [baseline_metrics.get('avg_loss', 0), ml_metrics.get('avg_loss', 0)]
        }, index=['Baseline', 'ML Enhanced'])
        
        win_loss_data.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('Win/Loss Comparison')
        axes[1, 2].set_ylabel('Average P&L ($)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('es_futures_ml_performance_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Performance report saved as 'es_futures_ml_performance_report.png'")

def main():
    pipeline = TradingSystemPipeline()
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print("\nThe ML-enhanced trading system has been successfully developed and tested.")
    print("All models, metrics, and visualizations have been generated.")
    
    return results

if __name__ == "__main__":
    results = main()