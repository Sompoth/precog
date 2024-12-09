from datetime import timedelta
from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import datetime_to_iso8601, iso8601_to_datetime


def get_point_estimate(timestamp: str) -> float:
    """Make a naive forecast by predicting the most recent price

    Args:
        timestamp (str): The current timestamp provided by the validator request

    Returns:
        (float): The current BTC price tied to the provided timestamp
    """
    # Create data gathering instance
    cm = CMData()

    # Set the time range to be as small as possible for query speed
    # Set the start time as 2 seconds prior to the provided time
    start_time: str = datetime_to_iso8601(iso8601_to_datetime(timestamp) - timedelta(seconds=2))
    end_time: str = timestamp

    # Query CM API for a pandas dataframe with only one record
    price_data: pd.DataFrame = cm.get_CM_ReferenceRate(assets="BTC", start=start_time, end=end_time)

    # Get current price closest to the provided timestamp
    btc_price: float = float(price_data["ReferenceRateUSD"].iloc[-1])

    # Return the current price of BTC as our point estimate
    return btc_price


def get_prediction_interval(timestamp: str, point_estimate: float) -> Tuple[float, float]:
    """Make a naive multi-step prediction interval by estimating
    the sample standard deviation

    Args:
        timestamp (str): The current timestamp provided by the validator request
        point_estimate (float): The center of the prediction interval

    Returns:
        (float): The 90% naive prediction interval lower bound
        (float): The 90% naive prediction interval upper bound

    Notes:
        Make reasonable assumptions that the 1s BTC price residuals are
        uncorrelated and normally distributed
    """
    # Create data gathering instance
    cm = CMData()

    # Set the time range to be 24 hours
    start_time: str = datetime_to_iso8601(iso8601_to_datetime(timestamp) - timedelta(days=1))
    end_time: str = timestamp

    # Query CM API for sample standard deviation of the 1s residuals
    historical_price_data: pd.DataFrame = cm.get_CM_ReferenceRate(
        assets="BTC", start=start_time, end=end_time, frequency="1s"
    )
    residuals: pd.Series = historical_price_data["ReferenceRateUSD"].diff()
    sample_std_dev: float = float(residuals.std())

    # We have the standard deviation of the 1s residuals
    # We are forecasting forward 5m, which is 300s
    # We must scale the 1s sample standard deviation to reflect a 300s forecast
    # Make reasonable assumptions that the 1s residuals are uncorrelated and normally distributed
    # To do this naively, we multiply the std dev by the square root of the number of time steps
    time_steps: int = 300
    naive_forecast_std_dev: float = sample_std_dev * (time_steps**0.5)

    # For a 90% prediction interval, we use the coefficient 1.64
    # Make reasonable assumptions that the 1s residuals are uncorrelated and normally distributed
    coefficient: float = 1.64

    # Calculate the lower bound and upper bound
    lower_bound: float = point_estimate - coefficient * naive_forecast_std_dev
    upper_bound: float = point_estimate + coefficient * naive_forecast_std_dev

    # Return the naive prediction interval for our forecast
    return lower_bound, upper_bound


async def forward(synapse: Challenge) -> Challenge:
    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
    )

    # Get the naive point estimate
    point_estimate: float = get_point_estimate(timestamp=synapse.timestamp)

    # Get the naive prediction interval
    prediction_interval: Tuple[float, float] = get_prediction_interval(
        timestamp=synapse.timestamp, point_estimate=point_estimate
    )

    synapse.prediction = point_estimate
    synapse.interval = prediction_interval

    if synapse.prediction is not None:
        bt.logging.success(f"Predicted price: {synapse.prediction}  |  Predicted Interval: {synapse.interval}")
    else:
        bt.logging.info("No prediction for this request.")
    return synapse
