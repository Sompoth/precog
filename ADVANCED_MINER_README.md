# Advanced Bitcoin Prediction Miner

This document describes the advanced prediction system for the Bittensor subnet 55 (Precog) miner.

## Overview

The advanced miner implements a sophisticated machine learning-based approach to Bitcoin price prediction, replacing the simple "current price" approach of the base miner with:

- **Multiple ML Models**: Random Forest, Gradient Boosting, Ridge Regression, and Linear Regression
- **Comprehensive Technical Indicators**: 50+ features including moving averages, RSI, MACD, Bollinger Bands, volatility measures, and more
- **Ensemble Predictions**: Combines multiple models for robust predictions
- **Advanced Volatility Modeling**: Sophisticated interval prediction using multiple volatility measures
- **Fallback Mechanisms**: Graceful degradation to simpler methods if advanced features fail

## Features

### Technical Indicators

The system creates 50+ technical indicators from Bitcoin price data:

#### Price-Based Features
- **Returns**: Percentage and logarithmic returns
- **Moving Averages**: SMA and EMA for multiple timeframes (5, 10, 20, 50, 100 periods)
- **Price Ratios**: Current price relative to various moving averages

#### Volatility Measures
- **Rolling Volatility**: Standard deviation of returns over multiple windows
- **Bollinger Bands**: Upper, lower bands and position within bands
- **Price Range**: High-low ranges over different periods

#### Momentum Indicators
- **RSI**: Relative Strength Index for 14 and 21 periods
- **MACD**: Moving Average Convergence Divergence with signal and histogram
- **Price Momentum**: Returns over various lag periods

#### Time-Based Features
- **Hour of Day**: Captures intraday patterns
- **Day of Week**: Weekly seasonality
- **Weekend Indicator**: Binary flag for weekend trading

#### Statistical Features
- **Lagged Values**: Price and returns at various lags
- **Rolling Statistics**: Skewness, kurtosis, min, max over different windows
- **Volume-Weighted**: VWAP calculations

### Machine Learning Models

#### Ensemble Approach
The system uses 4 different ML models and combines their predictions:

1. **Random Forest**: 100 trees, handles non-linear relationships
2. **Gradient Boosting**: 100 estimators, captures complex patterns
3. **Ridge Regression**: Regularized linear model, prevents overfitting
4. **Linear Regression**: Simple baseline model

#### Training Process
- Uses 7 days of historical data for training
- Features are standardized using StandardScaler
- Models predict price change (not absolute price) for better stability
- Ensemble prediction is the mean of all model predictions

### Prediction Intervals

#### Advanced Volatility Modeling
- **Multiple Volatility Measures**: Recent (1 hour), daily (24 hours), and overall period volatility
- **Weighted Combination**: 50% recent + 30% daily + 20% overall volatility
- **Time Scaling**: Properly scales 1-second volatility to 1-hour predictions
- **90% Confidence Intervals**: Uses 1.645 standard deviations

#### Fallback Method
If advanced volatility calculation fails, falls back to the simple method from base_miner.

## Usage

### Prerequisites

The advanced miner requires additional Python packages:

```bash
pip install scikit-learn numpy pandas
```

### Configuration

1. **Update .env.miner**:
   ```bash
   FORWARD_FUNCTION=advanced_miner
   ```

2. **Run the advanced miner**:
   ```bash
   make miner_advanced ENV_FILE=.env.miner
   ```

### Testing

Run the test script to validate the system:

```bash
python test_advanced_miner.py
```

This will:
- Create mock Bitcoin price data
- Test technical indicator creation
- Validate feature preparation
- Test the full prediction pipeline

## Performance Characteristics

### Speed
- **Training Time**: ~2-5 seconds (depends on data size)
- **Prediction Time**: ~0.1-0.5 seconds per request
- **Total Response Time**: ~1-3 seconds (including data fetching)

### Accuracy
- **Fallback Safety**: Always falls back to current price if advanced methods fail
- **Robust Error Handling**: Multiple layers of error handling and logging
- **Data Validation**: Checks for sufficient data before training

### Resource Usage
- **Memory**: ~100-200MB additional memory for ML models
- **CPU**: Moderate CPU usage during training, minimal during prediction
- **API Calls**: Same as base miner (uses CoinMetrics API)

## Architecture

### Class Structure

```
BitcoinPredictor
├── create_technical_indicators()  # Feature engineering
├── prepare_features()             # Data preparation
├── train_models()                 # Model training
└── predict_ensemble()             # Ensemble prediction

Main Functions
├── get_advanced_point_estimate()  # ML-based price prediction
├── get_advanced_prediction_interval()  # Volatility-based intervals
└── forward()                      # Main prediction interface
```

### Data Flow

1. **Data Fetching**: Get 7 days of 1-second Bitcoin price data from CoinMetrics
2. **Feature Engineering**: Create 50+ technical indicators
3. **Data Preparation**: Clean data, create target variables, handle missing values
4. **Model Training**: Train 4 ML models on historical data
5. **Prediction**: Use ensemble of models to predict price change
6. **Interval Calculation**: Use advanced volatility modeling for confidence intervals
7. **Fallback**: If any step fails, fall back to simpler methods

## Customization

### Adding New Features

To add new technical indicators, modify the `create_technical_indicators()` method:

```python
def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...
    
    # Add your custom indicator
    df['my_custom_indicator'] = your_calculation(df['price'])
    
    return df
```

### Adding New Models

To add new ML models, update the `models` dictionary in `BitcoinPredictor.__init__()`:

```python
self.models = {
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'ridge': Ridge(alpha=1.0),
    'linear': LinearRegression(),
    'your_model': YourModel()  # Add your model here
}
```

### Adjusting Prediction Horizon

Currently set to 1 hour. To change, modify the `target_hours` parameter in `prepare_features()` calls.

## Monitoring and Debugging

### Logging

The system provides detailed logging at different levels:

- **DEBUG**: Detailed timing and intermediate results
- **INFO**: High-level progress updates
- **WARNING**: Fallback activations and minor issues
- **ERROR**: Critical failures

### Performance Monitoring

Key metrics logged:
- Training time per model
- Prediction time
- Data quality metrics
- Fallback activations

### Common Issues

1. **Insufficient Data**: System needs at least 1000 data points for training
2. **API Failures**: Automatically falls back to current price
3. **Memory Issues**: Large datasets may require more RAM
4. **Model Training Failures**: Individual model failures don't stop the system

## Comparison with Base Miner

| Feature | Base Miner | Advanced Miner |
|---------|------------|----------------|
| Prediction Method | Current price | ML ensemble |
| Features | None | 50+ technical indicators |
| Models | None | 4 ML models |
| Interval Method | Simple volatility | Advanced volatility modeling |
| Training Data | None | 7 days historical |
| Fallback | None | Multiple fallback layers |
| Performance | ~0.1s | ~1-3s |
| Accuracy | Baseline | Potentially much higher |

## Future Improvements

### Potential Enhancements

1. **More Data Sources**: Incorporate additional market data
2. **Deep Learning**: Add neural networks for complex pattern recognition
3. **Online Learning**: Update models continuously with new data
4. **Feature Selection**: Automatically select most important features
5. **Hyperparameter Tuning**: Optimize model parameters
6. **Ensemble Weights**: Learn optimal weights for model combination

### Research Directions

1. **Market Regime Detection**: Adapt models to different market conditions
2. **Multi-Asset Features**: Incorporate other cryptocurrency data
3. **Sentiment Analysis**: Add news and social media sentiment
4. **Order Book Data**: Use market microstructure data
5. **Cross-Validation**: Implement proper time series cross-validation

## Conclusion

The advanced miner provides a significant upgrade over the base miner by incorporating sophisticated machine learning techniques and comprehensive technical analysis. While more complex, it maintains the same interface and includes robust fallback mechanisms to ensure reliability.

The system is designed to be easily extensible and customizable, allowing miners to experiment with different approaches and continuously improve their prediction accuracy.
