"""
Utility functions for time-series analysis.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd


def detect_seasonality(y: Union[np.ndarray, pd.Series], max_period: int = 52) -> int:
    """Detect the dominant seasonal period using autocorrelation.

    Parameters
    ----------
    y : array-like
        1D time series.
    max_period : int
        Maximum period to check.

    Returns
    -------
    int
        The most likely seasonal period (1 = no seasonality detected).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return 1

    y_mean = y.mean()
    y_centered = y - y_mean
    variance = (y_centered ** 2).mean()
    if variance == 0:
        return 1

    # Compute autocorrelation at each lag up to max_period
    max_lag = min(max_period, n // 3)
    acf = np.array([
        np.mean(y_centered[:n - k] * y_centered[k:]) / variance
        for k in range(1, max_lag + 1)
    ])

    # Find local maxima
    peaks = [
        i + 1  # lag (1-indexed)
        for i in range(1, len(acf) - 1)
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0
    ]
    if not peaks:
        return 1
    # Return the peak with the highest autocorrelation value
    return max(peaks, key=lambda lag: acf[lag - 1])


def train_test_split_ts(
    y: Union[np.ndarray, pd.Series],
    exog: Union[np.ndarray, pd.DataFrame, None] = None,
    test_frac: float = 0.2,
) -> tuple:
    """Chronological train/test split for time series.

    Parameters
    ----------
    y : array-like
        Target series.
    exog : array-like or None
        Optional exogenous matrix.
    test_frac : float
        Fraction of data for the test set [0, 1).

    Returns
    -------
    tuple
        ``(y_train, y_test)`` if *exog* is None, else
        ``(y_train, y_test, exog_train, exog_test)``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    split = n - max(1, int(n * test_frac))

    y_train, y_test = y[:split], y[split:]

    if exog is None:
        return y_train, y_test

    exog = np.asarray(exog)
    return y_train, y_test, exog[:split], exog[split:]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
