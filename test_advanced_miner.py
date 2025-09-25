#!/usr/bin/env python3
"""
Test script for the advanced miner prediction system.
This script tests the advanced miner without running the full Bittensor network.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from precog.miners.advanced_miner import BitcoinPredictor, get_advanced_point_estimate, get_advanced_prediction_interval
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str, to_datetime


def create_mock_data(days=7, frequency='1s'):
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
    daily_cycle = 0.01 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 3600))  # Daily cycle
    weekly_cycle = 0.02 * np.sin(2 * np.pi * np.arange(n_points) / (7 * 24 * 3600))  # Weekly cycle
    
    prices = base_price * (1 + trend + volatility + noise + daily_cycle + weekly_cycle)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'asset': 'BTC',
        'ReferenceRateUSD': prices
    })
    
    print(f"Created {len(df)} data points from {start_time} to {end_time}")
    return df


class MockCMData:
    """Mock CMData class for testing without API calls."""
    
    def __init__(self, mock_data):
        self.mock_data = mock_data
        self._cache = pd.DataFrame()
    
    def get_CM_ReferenceRate(self, assets="BTC", start=None, end=None, frequency="1s", 
                           limit_per_asset=None, paging_from="end", use_cache=False, **kwargs):
        """Mock the CM API call."""
        df = self.mock_data.copy()
        
        if start:
            start_time = pd.to_datetime(start)
            df = df[df['time'] >= start_time]
        
        if end:
            end_time = pd.to_datetime(end)
            df = df[df['time'] <= end_time]
        
        if limit_per_asset:
            if paging_from == "end":
                df = df.tail(limit_per_asset)
            else:
                df = df.head(limit_per_asset)
        
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


def test_full_prediction():
    """Test the full prediction pipeline."""
    print("\n=== Testing Full Prediction Pipeline ===")
    
    # Create mock data
    mock_data = create_mock_data(days=7, frequency='1min')
    mock_cm = MockCMData(mock_data)
    
    # Test timestamp
    test_timestamp = to_str(datetime.now())
    
    try:
        # Test point estimate
        print("Testing point estimate...")
        point_estimate = get_advanced_point_estimate(mock_cm, test_timestamp)
        print(f"Point estimate: ${point_estimate:,.2f}")
        
        # Test prediction interval
        print("Testing prediction interval...")
        lower, upper = get_advanced_prediction_interval(mock_cm, test_timestamp, point_estimate)
        print(f"Prediction interval: [${lower:,.2f}, ${upper:,.2f}]")
        print(f"Interval width: ${upper - lower:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"Full prediction test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Advanced Bitcoin Prediction System")
    print("=" * 50)
    
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
    
    # Test 3: Full prediction pipeline
    try:
        if test_full_prediction():
            print("✅ Full prediction pipeline test passed")
        else:
            print("❌ Full prediction pipeline test failed")
            return
    except Exception as e:
        print(f"❌ Full prediction pipeline test failed: {e}")
        return
    
    print("\n🎉 All tests passed! The advanced miner is ready to use.")
    print("\nTo use the advanced miner:")
    print("1. Update your .env.miner file with: FORWARD_FUNCTION=advanced_miner")
    print("2. Run: make miner_advanced ENV_FILE=.env.miner")


if __name__ == "__main__":
    main()
