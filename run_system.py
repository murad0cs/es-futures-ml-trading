#!/usr/bin/env python
"""
ES Futures ML Trading System - Quick Test Runner
This script provides a quick way to test the system with sample data
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.main import TradingSystemPipeline

def run_quick_test():
    """Run a quick test of the ML trading system"""
    print("\n" + "="*80)
    print(" ES FUTURES ML TRADING SYSTEM - QUICK TEST")
    print("="*80)
    print("\nThis will run a simplified test with synthetic data to verify the system works.")
    print("For production use, ensure you have proper market data access.\n")
    
    try:
        # Initialize pipeline
        pipeline = TradingSystemPipeline()
        
        # Override config for quick test
        pipeline.config.prediction_horizon = 10  # Shorter prediction horizon for speed
        pipeline.config.lookback_periods = 60    # Less lookback for speed
        
        # Run the complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print(" TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Summary of results
        if results:
            ml_metrics = results['ml_metrics']
            print("\nFinal ML System Metrics:")
            print(f"  Sharpe Ratio:  {ml_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Calmar Ratio:  {ml_metrics.get('calmar_ratio', 0):.3f}")
            print(f"  Profit Factor: {ml_metrics.get('profit_factor', 0):.3f}")
            print(f"  Win Rate:      {ml_metrics.get('win_rate', 0):.1f}%")
            print(f"  Max Drawdown:  {abs(ml_metrics.get('max_drawdown', 0)):.1f}%")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)