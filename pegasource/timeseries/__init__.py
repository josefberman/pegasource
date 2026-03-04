"""
pegasource.timeseries — Simple automatic time-series forecasting.

Supports univariate and multivariate (with exogenous variables) forecasting.
Uses SARIMAX (statsmodels) with automatic order selection.

Quick start::

    import numpy as np
    from pegasource.timeseries import AutoForecaster

    y = np.sin(np.linspace(0, 4 * np.pi, 48)) + np.random.randn(48) * 0.1
    fc = AutoForecaster()
    fc.fit(y)
    pred = fc.predict(steps=12)
    fc.plot(steps=12)
"""

from .auto import AutoForecaster
from .models import SARIMAXModel, LinearTrendModel
from .utils import detect_seasonality, train_test_split_ts, rmse

__all__ = [
    "AutoForecaster",
    "SARIMAXModel",
    "LinearTrendModel",
    "detect_seasonality",
    "train_test_split_ts",
    "rmse",
]
