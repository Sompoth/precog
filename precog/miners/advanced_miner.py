import time
from typing import Tuple

import bittensor as bt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


class BitcoinPredictor:
    """
    Advanced Bitcoin price prediction system using multiple ML models and technical indicators.
    """

    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "ridge": Ridge(alpha=1.0),
            "linear": LinearRegression(),
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators from price data.

        Args:
            df: DataFrame with 'ReferenceRateUSD' column and datetime index

        Returns:
            DataFrame with additional technical indicator columns
        """
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

        # Price-based features
        df["price"] = df["ReferenceRateUSD"]
        df["returns"] = df["price"].pct_change()
        df["log_returns"] = np.log(df["price"] / df["price"].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f"sma_{window}"] = df["price"].rolling(window=window).mean()
            df[f"ema_{window}"] = df["price"].ewm(span=window).mean()

        # Price ratios
        df["price_sma5_ratio"] = df["price"] / df["sma_5"]
        df["price_sma20_ratio"] = df["price"] / df["sma_20"]
        df["price_sma50_ratio"] = df["price"] / df["sma_50"]

        # Volatility measures
        for window in [5, 10, 20]:
            df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()
            df[f"volatility_{window}_log"] = df["log_returns"].rolling(window=window).std()

        # Bollinger Bands
        for window in [20, 50]:
            sma = df["price"].rolling(window=window).mean()
            std = df["price"].rolling(window=window).std()
            df[f"bb_upper_{window}"] = sma + (2 * std)
            df[f"bb_lower_{window}"] = sma - (2 * std)
            df[f"bb_position_{window}"] = (df["price"] - df[f"bb_lower_{window}"]) / (
                df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
            )

        # RSI (Relative Strength Index)
        for window in [14, 21]:
            delta = df["price"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["price"].ewm(span=12).mean()
        ema_26 = df["price"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Price momentum
        for window in [1, 5, 10, 20]:
            df[f"momentum_{window}"] = df["price"] / df["price"].shift(window) - 1

        # Volume-weighted features (using price as proxy for volume)
        for window in [10, 20]:
            df[f"vwap_{window}"] = (df["price"] * df["price"]).rolling(window=window).sum() / df["price"].rolling(
                window=window
            ).sum()

        # Time-based features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f"price_lag_{lag}"] = df["price"].shift(lag)
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20]:
            df[f"price_skew_{window}"] = df["returns"].rolling(window=window).skew()
            df[f"price_kurt_{window}"] = df["returns"].rolling(window=window).kurt()
            df[f"price_max_{window}"] = df["price"].rolling(window=window).max()
            df[f"price_min_{window}"] = df["price"].rolling(window=window).min()
            df[f"price_range_{window}"] = df[f"price_max_{window}"] - df[f"price_min_{window}"]

        return df

    def prepare_features(self, df: pd.DataFrame, target_hours: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training/prediction.

        Args:
            df: DataFrame with technical indicators
            target_hours: Hours ahead to predict

        Returns:
            Tuple of (features, target)
        """
        # Create target (price change in target_hours)
        target_hours_seconds = target_hours * 3600
        df["target"] = df["price"].shift(-target_hours_seconds) / df["price"] - 1

        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ["ReferenceRateUSD", "target", "time", "asset"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Remove rows with NaN values
        df_clean = df[feature_cols + ["target"]].dropna()

        if len(df_clean) == 0:
            raise ValueError("No valid data after cleaning")

        X = df_clean[feature_cols].values
        y = df_clean["target"].values

        self.feature_columns = feature_cols
        return X, y

    def train_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all models on the provided data.

        Args:
            X: Feature matrix
            y: Target values
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train each model
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                bt.logging.debug(f"Trained {name} model successfully")
            except Exception as e:
                bt.logging.warning(f"Failed to train {name} model: {e}")

        self.is_trained = True

    def predict_ensemble(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Make ensemble prediction with confidence interval.

        Args:
            X: Feature matrix for prediction

        Returns:
            Tuple of (prediction, confidence_std)
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
                bt.logging.warning(f"Failed to predict with {name}: {e}")

        if not predictions:
            raise ValueError("No models available for prediction")

        # Ensemble prediction (weighted average)
        ensemble_pred = np.mean(predictions)
        confidence_std = np.std(predictions)

        return ensemble_pred, confidence_std


def get_advanced_point_estimate(cm: CMData, timestamp: str) -> float:
    """
    Advanced point estimate using ML models and technical analysis.

    Args:
        cm: CoinMetrics API client
        timestamp: Current timestamp

    Returns:
        Predicted Bitcoin price
    """
    try:
        # Get historical data (last 7 days for training)
        end_time = to_datetime(timestamp)
        start_time = get_before(timestamp, days=7, minutes=0, seconds=0)

        # Fetch data
        historical_data = cm.get_CM_ReferenceRate(
            assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s", use_cache=True
        )

        if len(historical_data) < 1000:  # Need sufficient data
            bt.logging.warning("Insufficient historical data, falling back to current price")
            current_data = cm.get_CM_ReferenceRate(
                assets="BTC",
                start=None,
                end=to_str(end_time),
                frequency="1s",
                limit_per_asset=1,
                paging_from="end",
                use_cache=False,
            )
            return float(current_data["ReferenceRateUSD"].iloc[-1])

        # Create predictor and process data
        predictor = BitcoinPredictor()
        df_with_indicators = predictor.create_technical_indicators(historical_data)

        # Prepare features
        X, y = predictor.prepare_features(df_with_indicators, target_hours=1)

        if len(X) < 100:  # Need sufficient training data
            bt.logging.warning("Insufficient training data, falling back to current price")
            return float(historical_data["ReferenceRateUSD"].iloc[-1])

        # Train models
        predictor.train_models(X, y)

        # Get latest features for prediction
        latest_features = df_with_indicators.iloc[-1:][predictor.feature_columns].values

        # Make prediction
        price_change, _ = predictor.predict_ensemble(latest_features)
        current_price = float(historical_data["ReferenceRateUSD"].iloc[-1])
        predicted_price = current_price * (1 + price_change)

        bt.logging.debug(f"Advanced prediction: {predicted_price:.2f} (change: {price_change:.4f})")
        return predicted_price

    except Exception as e:
        bt.logging.warning(f"Advanced prediction failed: {e}, falling back to current price")
        # Fallback to current price
        current_data = cm.get_CM_ReferenceRate(
            assets="BTC",
            start=None,
            end=to_str(to_datetime(timestamp)),
            frequency="1s",
            limit_per_asset=1,
            paging_from="end",
            use_cache=False,
        )
        return float(current_data["ReferenceRateUSD"].iloc[-1])


def get_advanced_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    """
    Advanced prediction interval using volatility modeling and confidence intervals.

    Args:
        cm: CoinMetrics API client
        timestamp: Current timestamp
        point_estimate: Predicted price

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    try:
        # Get historical data for volatility analysis
        end_time = to_datetime(timestamp)
        start_time = get_before(timestamp, days=3, minutes=0, seconds=0)

        historical_data = cm.get_CM_ReferenceRate(
            assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s", use_cache=True
        )

        if len(historical_data) < 100:
            # Fallback to simple method
            return get_simple_interval(cm, timestamp, point_estimate)

        # Calculate returns and volatility
        df = historical_data.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df["returns"] = df["ReferenceRateUSD"].pct_change().dropna()

        # Multiple volatility measures
        recent_vol = df["returns"].tail(3600).std()  # Last hour
        daily_vol = df["returns"].tail(86400).std()  # Last day
        avg_vol = df["returns"].std()  # Overall period

        # Weighted volatility (more weight to recent)
        weighted_vol = 0.5 * recent_vol + 0.3 * daily_vol + 0.2 * avg_vol

        # Scale for 1-hour prediction
        forecast_vol = weighted_vol * np.sqrt(3600)  # 3600 seconds in 1 hour

        # Calculate confidence interval (90%)
        confidence_factor = 1.645  # 90% confidence
        margin = confidence_factor * forecast_vol * point_estimate

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        bt.logging.debug(f"Advanced interval: [{lower_bound:.2f}, {upper_bound:.2f}] (vol: {forecast_vol:.6f})")
        return lower_bound, upper_bound

    except Exception as e:
        bt.logging.warning(f"Advanced interval calculation failed: {e}, using simple method")
        return get_simple_interval(cm, timestamp, point_estimate)


def get_simple_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    """
    Fallback simple interval calculation (similar to base_miner).
    """
    start_time = get_before(timestamp, days=1, minutes=0, seconds=0)
    end_time = to_datetime(timestamp)

    historical_price_data = cm.get_CM_ReferenceRate(
        assets="BTC", start=to_str(start_time), end=to_str(end_time), frequency="1s"
    )

    residuals = historical_price_data["ReferenceRateUSD"].diff()
    sample_std_dev = float(residuals.std())

    time_steps = 3600
    naive_forecast_std_dev = sample_std_dev * (time_steps**0.5)
    coefficient = 1.64

    lower_bound = point_estimate - coefficient * naive_forecast_std_dev
    upper_bound = point_estimate + coefficient * naive_forecast_std_dev

    return lower_bound, upper_bound


def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """
    Advanced forward function with ML-based predictions.
    """
    total_start_time = time.perf_counter()
    bt.logging.info(f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}")

    try:
        # Get advanced point estimate
        point_estimate_start = time.perf_counter()
        point_estimate = get_advanced_point_estimate(cm=cm, timestamp=synapse.timestamp)
        point_estimate_time = time.perf_counter() - point_estimate_start
        bt.logging.debug(f"⏱️ Advanced point estimate took: {point_estimate_time:.3f} seconds")

        # Get advanced prediction interval
        interval_start = time.perf_counter()
        prediction_interval = get_advanced_prediction_interval(
            cm=cm, timestamp=synapse.timestamp, point_estimate=point_estimate
        )
        interval_time = time.perf_counter() - interval_start
        bt.logging.debug(f"⏱️ Advanced interval calculation took: {interval_time:.3f} seconds")

        # Set predictions
        synapse.prediction = point_estimate
        synapse.interval = list(prediction_interval)

        total_time = time.perf_counter() - total_start_time
        bt.logging.debug(f"⏱️ Total advanced forward call took: {total_time:.3f} seconds")

        if synapse.prediction is not None:
            bt.logging.success(f"Advanced prediction: {synapse.prediction:.2f} | Interval: {synapse.interval}")
        else:
            bt.logging.info("No prediction for this request.")

    except Exception as e:
        bt.logging.error(f"Advanced prediction failed: {e}")
        # Fallback to simple prediction
        from precog.miners.base_miner import get_point_estimate, get_prediction_interval

        synapse.prediction = get_point_estimate(cm, synapse.timestamp)
        synapse.interval = list(get_prediction_interval(cm, synapse.timestamp, synapse.prediction))
        bt.logging.info("Fell back to simple prediction method")

    return synapse
