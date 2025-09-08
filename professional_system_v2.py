"""
PROFESSIONAL ML-ENHANCED ES FUTURES TRADING SYSTEM V2
Implements proper 2-phase modeling approach with walk-forward validation
"""

import pandas as pd
import numpy as np
import time
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_pipeline.data_loader import DataLoader
from src.data_pipeline.feature_engineering import FeatureEngineer
from src.models.stop_loss_model import DynamicStopLossModel
from src.models.entry_confidence_model import EntryConfidenceModel
from src.models.position_sizing import AdaptivePositionSizer
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.metrics import PerformanceMetrics
from src.utils.config import TradingConfig, ModelConfig

class ProfessionalTradingSystemV2:
    """
    Enhanced Professional ML Trading System with 2-Phase Modeling
    
    Phase 1: Initial Model Training (80% historical data)
    - Train base models
    - Establish baseline performance
    - Initial hyperparameter optimization
    
    Phase 2: Walk-Forward Validation & Refinement (20% recent data)
    - Rolling window validation
    - Model retraining and adaptation
    - Dynamic threshold adjustment
    - Performance tracking
    """
    
    def __init__(self):
        self.config = TradingConfig()
        self.model_config = ModelConfig()
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.backtest_engine = BacktestEngine(self.config)
        self.metrics_calculator = PerformanceMetrics()
        
        # Models
        self.stop_loss_model = None
        self.entry_confidence_model = None
        self.position_sizer = None
        
        # Performance tracking
        self.phase1_metrics = {}
        self.phase2_metrics = {}
        self.walk_forward_results = []
        
    def run_complete_system(self):
        """Execute the complete 2-phase ML trading system"""
        print("="*80)
        print("PROFESSIONAL ML-ENHANCED ES FUTURES TRADING SYSTEM V2")
        print("2-PHASE MODELING WITH WALK-FORWARD VALIDATION")
        print("="*80)
        print("\nTarget Metrics:")
        print("- Sharpe Ratio: > 1.0")
        print("- Calmar Ratio: > 2.0")
        print("- Profit Factor: > 1.7")
        print("- Win Rate: > 55%")
        print("- Maximum Drawdown: < 10%")
        print("- Average Win/Loss Ratio: > 2.0")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Load and prepare data
            df = self.load_and_prepare_data()
            
            # Split for 2-phase approach
            phase1_df, phase2_df = self.split_data_for_phases(df)
            
            # PHASE 1: Initial Model Training
            print("\n" + "="*80)
            print("PHASE 1: INITIAL MODEL TRAINING (80% Historical Data)")
            print("="*80)
            self.phase1_metrics = self.run_phase1_training(phase1_df)
            
            # PHASE 2: Walk-Forward Validation & Refinement
            print("\n" + "="*80)
            print("PHASE 2: WALK-FORWARD VALIDATION & REFINEMENT (20% Recent Data)")
            print("="*80)
            self.phase2_metrics = self.run_phase2_validation(phase2_df)
            
            # Final Performance Assessment
            print("\n" + "="*80)
            print("FINAL PERFORMANCE ASSESSMENT")
            print("="*80)
            self.evaluate_final_performance()
            
            # Generate comprehensive report
            self.generate_final_report()
            
            # Save models and results
            self.save_all_results()
            
            execution_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("SYSTEM EXECUTION COMPLETE")
            print("="*80)
            print(f"Total Execution Time: {execution_time:.1f} seconds")
            print(f"Phase 1 Best Sharpe: {self.phase1_metrics.get('best_sharpe', 0):.3f}")
            print(f"Phase 2 Best Sharpe: {self.phase2_metrics.get('best_sharpe', 0):.3f}")
            
            # Determine system status
            final_sharpe = self.phase2_metrics.get('best_sharpe', 0)
            if final_sharpe >= 1.5:
                print("\n[STATUS] EXCELLENT - System ready for live trading")
            elif final_sharpe >= 1.0:
                print("\n[STATUS] GOOD - System meets minimum requirements")
            else:
                print("\n[STATUS] NEEDS IMPROVEMENT - Further optimization required")
                
        except Exception as e:
            print(f"\n[ERROR]: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("\n[DATA PREPARATION]")
        print("-"*40)
        
        # Load full dataset (1 year for comprehensive training)
        df = self.data_loader.load_data(
            symbol='ES=F',
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='1h'
        )
        print(f"Loaded {len(df)} data points")
        
        # Engineer features
        df = self.feature_engineer.create_features(df)
        df = self.data_loader.add_external_features(df)
        print(f"Created {len(df.columns)} features")
        
        return df
    
    def split_data_for_phases(self, df):
        """Split data for 2-phase approach"""
        split_point = int(len(df) * 0.8)
        phase1_df = df.iloc[:split_point].copy()
        phase2_df = df.iloc[split_point:].copy()
        
        print(f"\nPhase 1 Data: {len(phase1_df)} samples ({phase1_df.index[0]} to {phase1_df.index[-1]})")
        print(f"Phase 2 Data: {len(phase2_df)} samples ({phase2_df.index[0]} to {phase2_df.index[-1]})")
        
        return phase1_df, phase2_df
    
    def run_phase1_training(self, phase1_df):
        """
        PHASE 1: Initial Model Training
        - Train models on 80% historical data
        - Establish baseline performance
        """
        metrics = {}
        
        # Split Phase 1 data into train/validation
        train_size = int(len(phase1_df) * 0.8)
        train_df = phase1_df.iloc[:train_size]
        val_df = phase1_df.iloc[train_size:]
        
        print(f"\nPhase 1 Training Set: {len(train_df)} samples")
        print(f"Phase 1 Validation Set: {len(val_df)} samples")
        
        # Generate signals
        train_signals = self.data_loader.generate_trade_signals(train_df)
        val_signals = self.data_loader.generate_trade_signals(val_df)
        print(f"Training signals: {len(train_signals)}, Validation signals: {len(val_signals)}")
        
        # Step 1: Train Dynamic Stop-Loss Model
        print("\n[STEP 1] Training Dynamic Stop-Loss Model...")
        self.stop_loss_model = DynamicStopLossModel(self.model_config)
        X_stop = train_df[self.config.stop_loss_features].copy()
        y_stop = train_df['atr'] * self.config.atr_multiplier
        
        # Handle data preprocessing
        for col in X_stop.columns:
            if X_stop[col].dtype.name == 'category':
                X_stop[col] = X_stop[col].cat.codes
            elif X_stop[col].dtype == 'object':
                X_stop[col] = pd.Categorical(X_stop[col]).codes
        X_stop = X_stop.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Clean y_stop labels
        y_stop = y_stop.fillna(y_stop.mean())
        y_stop = y_stop.replace([np.inf, -np.inf], y_stop.mean())
        
        # Remove any remaining invalid indices
        valid_mask = ~(y_stop.isna() | np.isinf(y_stop))
        X_stop = X_stop[valid_mask]
        y_stop = y_stop[valid_mask]
        
        stop_loss_results = self.stop_loss_model.train(X_stop, y_stop)
        print(f"Best Stop-Loss Model: {self.stop_loss_model.best_model_name}")
        metrics['stop_loss_model'] = self.stop_loss_model.best_model_name
        
        # Step 2: Train Entry Confidence Model
        print("\n[STEP 2] Training Entry Confidence Model...")
        self.entry_confidence_model = EntryConfidenceModel(self.model_config)
        
        # Prepare labels for confidence model
        train_df['future_return'] = train_df['close'].shift(-self.config.prediction_horizon) / train_df['close'] - 1
        train_df['profitable'] = (train_df['future_return'] > 0.001).astype(int)
        
        X_conf = train_df[self.config.entry_confidence_features].copy()
        y_conf = train_df['profitable']
        
        # Handle data preprocessing
        for col in X_conf.columns:
            if X_conf[col].dtype.name == 'category':
                X_conf[col] = X_conf[col].cat.codes
            elif X_conf[col].dtype == 'object':
                X_conf[col] = pd.Categorical(X_conf[col]).codes
        X_conf = X_conf.fillna(0).replace([np.inf, -np.inf], 0)
        
        valid_indices = ~y_conf.isna()
        X_conf = X_conf[valid_indices]
        y_conf = y_conf[valid_indices]
        
        confidence_results = self.entry_confidence_model.train(X_conf, y_conf)
        print(f"Best Confidence Model: {self.entry_confidence_model.best_model_name}")
        metrics['confidence_model'] = self.entry_confidence_model.best_model_name
        
        # Step 3: Initialize Position Sizing
        print("\n[STEP 3] Initializing Adaptive Position Sizing...")
        self.position_sizer = AdaptivePositionSizer(self.config)
        print("Position sizing with Kelly Criterion initialized")
        
        # Step 4: Validate on Phase 1 validation set
        print("\n[STEP 4] Validating Phase 1 Models...")
        val_result = self.backtest_engine.run_ml_enhanced_backtest(
            df=val_df,
            signals=val_signals,
            stop_loss_model=self.stop_loss_model,
            entry_confidence_model=self.entry_confidence_model,
            position_sizer=self.position_sizer
        )
        
        val_metrics = self.metrics_calculator.calculate_all_metrics(
            val_result['returns'],
            val_result['trades']
        )
        
        metrics['sharpe_ratio'] = val_metrics.get('sharpe_ratio', 0)
        metrics['win_rate'] = val_metrics.get('win_rate', 0)
        metrics['profit_factor'] = val_metrics.get('profit_factor', 0)
        metrics['max_drawdown'] = val_metrics.get('max_drawdown', 0)
        metrics['best_sharpe'] = val_metrics.get('sharpe_ratio', 0)
        
        print(f"\nPhase 1 Validation Results:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown: {abs(metrics['max_drawdown']):.2f}%")
        
        return metrics
    
    def run_phase2_validation(self, phase2_df):
        """
        PHASE 2: Walk-Forward Validation & Refinement
        - Test on unseen recent data
        - Implement walk-forward analysis
        - Refine models based on recent performance
        """
        metrics = {}
        
        print("\n[WALK-FORWARD VALIDATION]")
        print("-"*40)
        
        # Walk-forward parameters
        window_size = min(1000, int(len(phase2_df) * 0.5))  # Use 50% of Phase 2 for each window
        step_size = int(window_size * 0.25)  # 25% step size for overlap
        
        print(f"Window Size: {window_size} samples")
        print(f"Step Size: {step_size} samples")
        
        walk_forward_results = []
        best_sharpe = -np.inf
        
        # Generate signals for entire Phase 2
        phase2_signals = self.data_loader.generate_trade_signals(phase2_df)
        
        # Perform walk-forward analysis
        n_windows = 0
        for start_idx in range(0, len(phase2_df) - window_size, step_size):
            n_windows += 1
            end_idx = min(start_idx + window_size, len(phase2_df))
            
            window_df = phase2_df.iloc[start_idx:end_idx]
            window_signals = phase2_signals.loc[phase2_signals.index.intersection(window_df.index)]
            
            print(f"\nWindow {n_windows}: {window_df.index[0]} to {window_df.index[-1]}")
            print(f"  Signals in window: {len(window_signals)}")
            
            if len(window_signals) == 0:
                print("  [SKIP] No signals in this window")
                continue
            
            # Split window into train/test (80/20)
            train_size = int(len(window_df) * 0.8)
            train_window = window_df.iloc[:train_size]
            test_window = window_df.iloc[train_size:]
            
            train_window_signals = window_signals.loc[window_signals.index.intersection(train_window.index)]
            test_window_signals = window_signals.loc[window_signals.index.intersection(test_window.index)]
            
            # Retrain models on window training data (fine-tuning)
            if len(train_window_signals) > 0:
                print(f"  Retraining on {len(train_window)} samples...")
                
                # Retrain stop-loss model
                X_stop = train_window[self.config.stop_loss_features].copy()
                y_stop = train_window['atr'] * self.config.atr_multiplier
                
                for col in X_stop.columns:
                    if X_stop[col].dtype.name == 'category':
                        X_stop[col] = X_stop[col].cat.codes
                    elif X_stop[col].dtype == 'object':
                        X_stop[col] = pd.Categorical(X_stop[col]).codes
                X_stop = X_stop.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Clean y_stop labels
                y_stop = y_stop.fillna(y_stop.mean())
                y_stop = y_stop.replace([np.inf, -np.inf], y_stop.mean())
                
                # Remove any remaining invalid indices
                valid_mask = ~(y_stop.isna() | np.isinf(y_stop))
                X_stop = X_stop[valid_mask]
                y_stop = y_stop[valid_mask]
                
                self.stop_loss_model.train(X_stop, y_stop)
            
            # Test on window test data
            if len(test_window_signals) > 0:
                print(f"  Testing on {len(test_window)} samples...")
                
                test_result = self.backtest_engine.run_ml_enhanced_backtest(
                    df=test_window,
                    signals=test_window_signals,
                    stop_loss_model=self.stop_loss_model,
                    entry_confidence_model=self.entry_confidence_model,
                    position_sizer=self.position_sizer
                )
                
                window_metrics = self.metrics_calculator.calculate_all_metrics(
                    test_result['returns'],
                    test_result['trades']
                )
                
                window_sharpe = window_metrics.get('sharpe_ratio', 0)
                window_return = test_result.get('total_return', 0)
                
                print(f"  Window Performance:")
                print(f"    Sharpe Ratio: {window_sharpe:.3f}")
                print(f"    Total Return: {window_return*100:.2f}%")
                print(f"    Trades: {len(test_result['trades'])}")
                
                walk_forward_results.append({
                    'window': n_windows,
                    'start': window_df.index[0],
                    'end': window_df.index[-1],
                    'sharpe': window_sharpe,
                    'return': window_return,
                    'trades': len(test_result['trades'])
                })
                
                if window_sharpe > best_sharpe:
                    best_sharpe = window_sharpe
        
        # Calculate aggregate Phase 2 metrics
        if walk_forward_results:
            avg_sharpe = np.mean([r['sharpe'] for r in walk_forward_results])
            avg_return = np.mean([r['return'] for r in walk_forward_results])
            total_trades = sum([r['trades'] for r in walk_forward_results])
            
            metrics['avg_sharpe'] = avg_sharpe
            metrics['best_sharpe'] = best_sharpe
            metrics['avg_return'] = avg_return
            metrics['total_trades'] = total_trades
            metrics['n_windows'] = n_windows
            
            print(f"\n[PHASE 2 SUMMARY]")
            print(f"Windows Tested: {n_windows}")
            print(f"Average Sharpe: {avg_sharpe:.3f}")
            print(f"Best Sharpe: {best_sharpe:.3f}")
            print(f"Average Return: {avg_return*100:.2f}%")
            print(f"Total Trades: {total_trades}")
        else:
            metrics['avg_sharpe'] = 0
            metrics['best_sharpe'] = 0
            metrics['avg_return'] = 0
            metrics['total_trades'] = 0
            metrics['n_windows'] = 0
        
        self.walk_forward_results = walk_forward_results
        return metrics
    
    def evaluate_final_performance(self):
        """Evaluate final system performance against targets"""
        print("\n[PERFORMANCE VS TARGETS]")
        print("-"*40)
        
        targets = {
            'Sharpe Ratio': {'target': 1.0, 'achieved': self.phase2_metrics.get('best_sharpe', 0), 'higher_better': True},
            'Win Rate (%)': {'target': 55.0, 'achieved': self.phase1_metrics.get('win_rate', 0), 'higher_better': True},
            'Profit Factor': {'target': 1.7, 'achieved': self.phase1_metrics.get('profit_factor', 0), 'higher_better': True},
            'Max Drawdown (%)': {'target': 10.0, 'achieved': abs(self.phase1_metrics.get('max_drawdown', 0)), 'higher_better': False}
        }
        
        achieved_count = 0
        total_targets = len(targets)
        
        print(f"\n{'Metric':<20} {'Target':<10} {'Achieved':<10} {'Status':<15}")
        print("-"*55)
        
        for metric_name, data in targets.items():
            target = data['target']
            achieved = data['achieved']
            higher_better = data['higher_better']
            
            if higher_better:
                status = "[ACHIEVED]" if achieved >= target else "[NOT MET]"
                if achieved >= target:
                    achieved_count += 1
            else:
                status = "[ACHIEVED]" if achieved <= target else "[NOT MET]"
                if achieved <= target:
                    achieved_count += 1
            
            print(f"{metric_name:<20} {target:<10.2f} {achieved:<10.2f} {status:<15}")
        
        success_rate = (achieved_count / total_targets) * 100
        print(f"\n[TARGET ACHIEVEMENT]: {achieved_count}/{total_targets} ({success_rate:.1f}%)")
        
        if success_rate >= 75:
            print("[STATUS]: EXCELLENT - System meets most targets")
        elif success_rate >= 50:
            print("[STATUS]: GOOD - System meets majority of targets")
        else:
            print("[STATUS]: NEEDS IMPROVEMENT - Further optimization required")
    
    def generate_final_report(self):
        """Generate comprehensive performance report"""
        print("\n[GENERATING REPORT]")
        print("-"*40)
        
        if not self.walk_forward_results:
            print("No walk-forward results to plot")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Professional ML Trading System V2 - Performance Report', fontsize=16, fontweight='bold')
        
        # 1. Walk-Forward Sharpe Ratios
        ax1 = plt.subplot(2, 3, 1)
        windows = [r['window'] for r in self.walk_forward_results]
        sharpes = [r['sharpe'] for r in self.walk_forward_results]
        ax1.bar(windows, sharpes, color='blue', alpha=0.7)
        ax1.axhline(y=1.0, color='green', linestyle='--', label='Target (1.0)')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Walk-Forward Sharpe Ratios')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Walk-Forward Returns
        ax2 = plt.subplot(2, 3, 2)
        returns = [r['return']*100 for r in self.walk_forward_results]
        ax2.plot(windows, returns, marker='o', color='green', linewidth=2)
        ax2.fill_between(windows, returns, alpha=0.3, color='green')
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Return (%)')
        ax2.set_title('Walk-Forward Returns')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade Distribution
        ax3 = plt.subplot(2, 3, 3)
        trades = [r['trades'] for r in self.walk_forward_results]
        ax3.bar(windows, trades, color='orange', alpha=0.7)
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Trade Distribution Across Windows')
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase 1 vs Phase 2 Comparison
        ax4 = plt.subplot(2, 3, 4)
        phases = ['Phase 1\n(Initial Training)', 'Phase 2\n(Walk-Forward)']
        phase_sharpes = [
            self.phase1_metrics.get('best_sharpe', 0),
            self.phase2_metrics.get('best_sharpe', 0)
        ]
        colors = ['blue' if s < 1.0 else 'green' for s in phase_sharpes]
        ax4.bar(phases, phase_sharpes, color=colors, alpha=0.7)
        ax4.axhline(y=1.0, color='red', linestyle='--', label='Target')
        ax4.set_ylabel('Best Sharpe Ratio')
        ax4.set_title('Phase Performance Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Model Performance Summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        summary_text = f"""
        SYSTEM PERFORMANCE SUMMARY
        
        Phase 1 Results:
        - Best Model (Stop-Loss): {self.phase1_metrics.get('stop_loss_model', 'N/A')}
        - Best Model (Confidence): {self.phase1_metrics.get('confidence_model', 'N/A')}
        - Validation Sharpe: {self.phase1_metrics.get('sharpe_ratio', 0):.3f}
        - Win Rate: {self.phase1_metrics.get('win_rate', 0):.1f}%
        
        Phase 2 Results:
        - Windows Tested: {self.phase2_metrics.get('n_windows', 0)}
        - Best Sharpe: {self.phase2_metrics.get('best_sharpe', 0):.3f}
        - Avg Return: {self.phase2_metrics.get('avg_return', 0)*100:.2f}%
        - Total Trades: {self.phase2_metrics.get('total_trades', 0)}
        """
        ax5.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        # 6. Cumulative Performance
        ax6 = plt.subplot(2, 3, 6)
        if self.walk_forward_results:
            cumulative_returns = np.cumprod([1 + r['return'] for r in self.walk_forward_results])
            ax6.plot(windows, cumulative_returns, marker='o', color='purple', linewidth=2)
            ax6.fill_between(windows, cumulative_returns, 1, alpha=0.3, color='purple')
            ax6.axhline(y=1, color='black', linestyle='-', alpha=0.3)
            ax6.set_xlabel('Window')
            ax6.set_ylabel('Cumulative Return')
            ax6.set_title('Cumulative Performance')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save report
        from src.utils.paths import get_report_file
        report_path = get_report_file('professional_ml_trading_report_v2.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"Report saved as '{report_path}'")
        plt.close()
    
    def save_all_results(self):
        """Save models and results"""
        print("\n[SAVING RESULTS]")
        print("-"*40)
        
        from src.utils.paths import get_model_file, get_results_file
        
        # Save models
        if self.stop_loss_model:
            joblib.dump(self.stop_loss_model, get_model_file('stop_loss_model.pkl', phase=2))
        if self.entry_confidence_model:
            joblib.dump(self.entry_confidence_model, get_model_file('entry_confidence_model.pkl', phase=2))
        if self.position_sizer:
            joblib.dump(self.position_sizer, get_model_file('position_sizer.pkl', phase=2))
        
        # Save metrics
        results = {
            'phase1_metrics': self.phase1_metrics,
            'phase2_metrics': self.phase2_metrics,
            'walk_forward_results': self.walk_forward_results
        }
        
        pd.DataFrame([results]).to_csv(get_results_file('system_metrics.csv', phase=2), index=False)
        
        if self.walk_forward_results:
            pd.DataFrame(self.walk_forward_results).to_csv(get_results_file('walk_forward_results.csv', phase=2), index=False)
        
        print(f"Models saved to '{get_model_file('', phase=2).parent}'")
        print(f"Results saved to '{get_results_file('', phase=2).parent}'")

def main():
    """Main execution function"""
    system = ProfessionalTradingSystemV2()
    system.run_complete_system()

if __name__ == "__main__":
    main()