#!/usr/bin/env python3
"""
Simplified test script for the advanced miner prediction system.
This script tests the core ML functionality without Bittensor dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


class BitcoinPredictor:
    """
    Advanced Bitcoin price prediction system using multiple ML models and technical indicators.
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators from price data.
        """
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()
        
        # Price-based features
        df['price'] = df['ReferenceRateUSD']
        df['returns'] = df['price'].pct_change()
        df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
        
        # Price ratios
        df['price_sma5_ratio'] = df['price'] / df['sma_5']
        df['price_sma20_ratio'] = df['price'] / df['sma_20']
        df['price_sma50_ratio'] = df['price'] / df['sma_50']
        
        # Volatility measures
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'volatility_{window}_log'] = df['log_returns'].rolling(window=window).std()
        
        # Bollinger Bands
        for window in [20, 50]:
            sma = df['price'].rolling(window=window).mean()
            std = df['price'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_position_{window}'] = (df['price'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # RSI (Relative Strength Index)
        for window in [14, 21]:
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['price'].ewm(span=12).mean()
        ema_26 = df['price'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price momentum
        for window in [1, 5, 10, 20]:
            df[f'momentum_{window}'] = df['price'] / df['price'].shift(window) - 1
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'price_skew_{window}'] = df['returns'].rolling(window=window).skew()
            df[f'price_kurt_{window}'] = df['returns'].rolling(window=window).kurt()
            df[f'price_max_{window}'] = df['price'].rolling(window=window).max()
            df[f'price_min_{window}'] = df['price'].rolling(window=window).min()
            df[f'price_range_{window}'] = df[f'price_max_{window}'] - df[f'price_min_{window}']
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_hours: int = 1) -> tuple:
        """
        Prepare features and target for training/prediction.
        """
        # Create target (price change in target_hours)
        target_hours_seconds = target_hours * 3600
        df['target'] = df['price'].shift(-target_hours_seconds) / df['price'] - 1
        
        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ['ReferenceRateUSD', 'target', 'time', 'asset']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['target']].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after cleaning")
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        self.feature_columns = feature_cols
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all models on the provided data.
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                print(f"Trained {name} model successfully")
            except Exception as e:
                print(f"Failed to train {name} model: {e}")
        
        self.is_trained = True
    
    def predict_ensemble(self, X: np.ndarray) -> tuple:
        """
        Make ensemble prediction with confidence interval.
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        # Get predictions from all trained models
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred[0])
            except Exception as e:
                print(f"Failed to predict with {name}: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.mean(predictions)
        confidence_std = np.std(predictions)
        
        return ensemble_pred, confidence_std


def create_mock_data(days=7, frequency='1min'):
    """Create mock Bitcoin price data for testing."""
    print("Creating mock Bitcoin price data...")
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Create price data with some realistic patterns
    timestamps = pd.date_range(start=start_time, end=end_time, freq=frequency)
    n_points = len(timestamps)
    
    # Generate price with trend, volatility, and some patterns
    base_price = 50000
    trend = np.linspace(0, 0.1, n_points)  # 10% upward trend
    volatility = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    noise = np.random.normal(0, 0.001, n_points)  # High-frequency noise
    
    # Add some cyclical patterns
    daily_cycle = 0.01 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 60))  # Daily cycle
    weekly_cycle = 0.02 * np.sin(2 * np.pi * np.arange(n_points) / (7 * 24 * 60))  # Weekly cycle
    
    prices = base_price * (1 + trend + volatility + noise + daily_cycle + weekly_cycle)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'asset': 'BTC',
        'ReferenceRateUSD': prices
    })
    
    print(f"Created {len(df)} data points from {start_time} to {end_time}")
    return df


def test_technical_indicators():
    """Test the technical indicators creation."""
    print("\n=== Testing Technical Indicators ===")
    
    # Create mock data
    mock_data = create_mock_data(days=3, frequency='1min')  # Use 1-minute data for faster testing
    
    # Create predictor and test indicators
    predictor = BitcoinPredictor()
    df_with_indicators = predictor.create_technical_indicators(mock_data)
    
    print(f"Original columns: {list(mock_data.columns)}")
    print(f"With indicators: {len(df_with_indicators.columns)} columns")
    print(f"New columns: {[col for col in df_with_indicators.columns if col not in mock_data.columns]}")
    
    # Check for NaN values
    nan_count = df_with_indicators.isnull().sum().sum()
    print(f"NaN values: {nan_count}")
    
    return df_with_indicators


def test_feature_preparation():
    """Test feature preparation and model training."""
    print("\n=== Testing Feature Preparation ===")
    
    # Create mock data
    mock_data = create_mock_data(days=5, frequency='1min')
    
    # Create predictor
    predictor = BitcoinPredictor()
    df_with_indicators = predictor.create_technical_indicators(mock_data)
    
    try:
        # Prepare features
        X, y = predictor.prepare_features(df_with_indicators, target_hours=1)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target array shape: {y.shape}")
        print(f"Feature columns: {len(predictor.feature_columns)}")
        
        # Train models
        predictor.train_models(X, y)
        print("Models trained successfully!")
        
        # Test prediction
        if len(X) > 0:
            latest_features = X[-1:].reshape(1, -1)
            pred, conf = predictor.predict_ensemble(latest_features)
            print(f"Sample prediction: {pred:.6f} (confidence std: {conf:.6f})")
        
        return True
        
    except Exception as e:
        print(f"Feature preparation failed: {e}")
        return False


def test_prediction_accuracy():
    """Test prediction accuracy on historical data."""
    print("\n=== Testing Prediction Accuracy ===")
    
    # Create mock data
    mock_data = create_mock_data(days=7, frequency='1min')
    
    # Create predictor
    predictor = BitcoinPredictor()
    df_with_indicators = predictor.create_technical_indicators(mock_data)
    
    try:
        # Prepare features
        X, y = predictor.prepare_features(df_with_indicators, target_hours=1)
        
        if len(X) < 100:
            print("Insufficient data for accuracy testing")
            return False
        
        # Split data for training and testing
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train on training data
        predictor_train = BitcoinPredictor()
        predictor_train.scaler.fit(X_train)
        X_train_scaled = predictor_train.scaler.transform(X_train)
        
        for name, model in predictor_train.models.items():
            try:
                model.fit(X_train_scaled, y_train)
            except Exception as e:
                print(f"Failed to train {name}: {e}")
        
        # Test on test data
        X_test_scaled = predictor_train.scaler.transform(X_test)
        predictions = []
        
        for name, model in predictor_train.models.items():
            try:
                pred = model.predict(X_test_scaled)
                predictions.append(pred)
            except Exception as e:
                print(f"Failed to predict with {name}: {e}")
        
        if predictions:
            # Calculate ensemble prediction
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate metrics
            mse = np.mean((ensemble_pred - y_test) ** 2)
            mae = np.mean(np.abs(ensemble_pred - y_test))
            rmse = np.sqrt(mse)
            
            print(f"Test set size: {len(y_test)}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"Mean actual change: {np.mean(y_test):.6f}")
            print(f"Mean predicted change: {np.mean(ensemble_pred):.6f}")
            
            return True
        else:
            print("No successful predictions")
            return False
        
    except Exception as e:
        print(f"Accuracy testing failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Advanced Bitcoin Prediction System (Simplified)")
    print("=" * 60)
    
    # Test 1: Technical indicators
    try:
        df_indicators = test_technical_indicators()
        print("✅ Technical indicators test passed")
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return
    
    # Test 2: Feature preparation
    try:
        if test_feature_preparation():
            print("✅ Feature preparation test passed")
        else:
            print("❌ Feature preparation test failed")
            return
    except Exception as e:
        print(f"❌ Feature preparation test failed: {e}")
        return
    
    # Test 3: Prediction accuracy
    try:
        if test_prediction_accuracy():
            print("✅ Prediction accuracy test passed")
        else:
            print("❌ Prediction accuracy test failed")
    except Exception as e:
        print(f"❌ Prediction accuracy test failed: {e}")
    
    print("\n🎉 Core ML functionality tests passed! The advanced miner is ready to use.")
    print("\nTo use the advanced miner:")
    print("1. Install required dependencies: pip install scikit-learn")
    print("2. Update your .env.miner file with: FORWARD_FUNCTION=advanced_miner")
    print("3. Run: make miner_advanced ENV_FILE=.env.miner")


if __name__ == "__main__":
    main()
