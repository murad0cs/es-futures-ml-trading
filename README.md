# ES Futures ML Trading System

## Overview
Advanced machine learning system for ES futures trading that enhances a static ATR-based algorithm with dynamic risk management and intelligent trade filtering.

## Key Features

### Phase 1: Dynamic Stop-Loss Model
- Predicts optimal stop-loss prices based on market volatility
- Uses ensemble of XGBoost, LightGBM, and Neural Networks
- Target: Improve Average Win/Loss Ratio > 2.0

### Phase 2: Entry Confidence Score
- Filters low-quality trade setups using ML classification
- Probabilistic confidence scoring with calibration
- Target: Win Rate > 55%, Profit Factor > 1.7

### Phase 3: Adaptive Position Sizing
- Dynamic position sizing based on volatility and confidence
- Kelly Criterion optimization
- Target: Max Drawdown < 10%

## Installation

```bash
# Clone the repository
cd es_futures_ml_trading

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the System
```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the ML-enhanced trading system
python professional_system_v2.py
```

### System Output
The system will:
1. Load and prepare data (synthetic or real)
2. Train models in Phase 1 (80% historical data)
3. Perform walk-forward validation in Phase 2 (20% recent data)
4. Generate comprehensive performance report
5. Save trained models and results

## Project Structure
```
es_futures_ml_trading/
├── professional_system_v2.py    # Main system entry point
├── src/
│   ├── data_pipeline/           # Data loading and feature engineering
│   │   ├── data_loader.py      # Data ingestion and signal generation
│   │   └── feature_engineering.py  # 85+ technical indicators
│   ├── models/                  # ML models
│   │   ├── stop_loss_model.py  # Dynamic stop-loss prediction
│   │   └── entry_confidence_model.py  # Trade quality scoring
│   ├── backtesting/             # Backtesting framework
│   │   ├── backtest_engine.py  # ML-enhanced backtesting
│   │   ├── performance_metrics.py  # Comprehensive metrics
│   │   └── position_sizing.py  # Adaptive position sizing
│   └── utils/                   # Utilities
│       ├── config.py            # System configuration
│       └── paths.py             # Centralized path management
├── data/                        # Market data storage
├── models/                      # Saved ML models
├── results/                     # Backtest results
├── reports/                     # Performance reports
├── requirements.txt             # Project dependencies
├── README.md                    # Documentation
└── IMPLEMENTATION_REVIEW.md     # Detailed implementation review
```

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted returns |
| Calmar Ratio | > 2.0 | Return over max drawdown |
| Profit Factor | > 1.7 | Gross profit / gross loss |
| Win Rate | > 55% | Percentage of winning trades |
| Max Drawdown | < 10% | Maximum peak-to-trough decline |
| Win/Loss Ratio | > 2.0 | Average win / average loss |

## Features Engineering

### Price Features
- Returns (multiple timeframes)
- Price momentum indicators
- Pullback depth and speed
- Support/resistance distances

### Volume Features
- Volume ratios and surges
- VWAP calculations
- Order flow imbalance
- On-balance volume

### Volatility Features
- ATR (Average True Range)
- Historical volatility (multiple periods)
- Parkinson and Garman-Klass volatility
- Volatility regime classification

### Technical Indicators
- RSI, MACD, Stochastic
- Bollinger Bands
- ADX trend strength
- Moving averages

### Market Microstructure
- Tick distribution
- Liquidity scoring
- Price efficiency metrics
- Order flow analysis

## Model Architecture

### Stop-Loss Models
- **XGBoost**: Gradient boosting with Optuna hyperparameter optimization
- **LightGBM**: Fast gradient boosting with categorical features
- **Neural Network**: Deep learning with batch normalization and dropout
- **Ensemble**: Meta-learning combination of all models

### Entry Confidence Models
- Binary classification for trade quality
- Probability calibration for reliable confidence scores
- Class weighting for imbalanced data
- ROC-AUC optimization

### Position Sizing
- Kelly Criterion for optimal sizing
- Volatility-based adjustments
- Portfolio heat management
- Correlation-based position adjustment

## Backtesting Framework

### Features
- Walk-forward analysis
- Transaction costs and slippage modeling
- Multiple exit strategies
- Comprehensive metrics calculation

### Metrics Calculated
- Return metrics (total, annual, cumulative)
- Risk metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis (max DD, recovery time, Ulcer Index)
- Trade statistics (win rate, profit factor, consecutive wins/losses)
- Distribution metrics (skewness, kurtosis, gain-to-pain)

## Results Analysis

The system generates comprehensive reports including:
1. **Equity curve comparison** (Baseline vs ML-enhanced)
2. **Drawdown analysis**
3. **Trade distribution histograms**
4. **Key metrics comparison charts**
5. **Rolling volatility analysis**
6. **Win/loss comparison**

## Model Training

The system uses:
- **Optuna** for hyperparameter optimization
- **Cross-validation** for robust model selection
- **Early stopping** to prevent overfitting
- **Stratified splits** for balanced training

## Risk Management

### Portfolio Heat Control
- Maximum 6% portfolio heat at any time
- Dynamic position limits based on current exposure
- Correlation-adjusted position sizing

### Stop-Loss Management
- Volatility-adaptive stop distances
- Regime-based multiplier adjustments
- Smoothing for stability

## Performance Report

After execution, the system generates:
- Detailed performance metrics comparison
- Visual analysis charts (saved as PNG)
- Trade-by-trade analysis
- Achievement status for all target metrics

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost, lightgbm
- tensorflow/keras
- optuna
- matplotlib, seaborn
- yfinance (for market data)

## Current Status

### Fully Implemented
- All three ML models (stop-loss, confidence, position sizing)
- Two-phase modeling approach with walk-forward validation
- Comprehensive feature engineering (85+ indicators)
- Complete backtesting framework with metrics
- Hyperparameter optimization using Optuna
- Professional code structure with modular design

### Known Limitations
- Currently uses synthetic data (real ES futures data needed for production)
- ML models are conservative due to synthetic data training
- Performance metrics require real market data for validation

### To Achieve Target Metrics
1. Replace synthetic data with real ES futures 1-minute data
2. Retrain models on actual market patterns
3. Fine-tune confidence thresholds based on real results
4. Iterate on feature engineering with domain expertise

## Notes

- The system uses synthetic data if real market data is unavailable
- All models are automatically tuned using Bayesian optimization
- Backtesting includes realistic transaction costs (0.05% commission, 0.01% slippage)
- The system is designed for 1-minute ES futures data but currently runs on hourly for testing
- Models use ensemble methods to reduce overfitting

## License

This project is for educational and research purposes only. Not financial advice.

## Author

/murad0cs

## Contact

For questions or improvements, please open an issue in the repository.