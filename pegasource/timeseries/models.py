"""
Model wrappers for time-series forecasting.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


class SARIMAXModel:
    """Thin wrapper around statsmodels SARIMAX.

    Parameters
    ----------
    order : tuple[int, int, int]
        ARIMA ``(p, d, q)`` order.
    seasonal_order : tuple[int, int, int, int]
        Seasonal ARIMA ``(P, D, Q, s)`` order.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> "SARIMAXModel":
        """Fit the SARIMAX model.

        Parameters
        ----------
        y : array-like
            Endogenous time series.
        exog : array-like or None
            Exogenous regressors.

        Returns
        -------
        self
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

        model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, warn_convergence=False)
        return self

    def predict(
        self,
        steps: int,
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> pd.Series:
        """Produce out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Number of steps ahead.
        exog : array-like or None
            Exogenous values for forecast horizon (required if model was fitted
            with exogenous variables).

        Returns
        -------
        pd.Series
        """
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        fc = self._result.forecast(steps=steps, exog=exog)
        return fc

    @property
    def aic(self) -> float:
        """Akaike Information Criterion (lower = better)."""
        if self._result is None:
            return float("inf")
        return float(self._result.aic)

    @property
    def in_sample_predictions(self) -> pd.Series:
        """In-sample fitted values."""
        return self._result.fittedvalues

    @property
    def residuals(self) -> pd.Series:
        """Model residuals."""
        return self._result.resid


class LinearTrendModel:
    """Simple OLS linear trend + seasonal dummies fallback model.

    Parameters
    ----------
    period : int
        Seasonal period (1 = no seasonality).
    """

    def __init__(self, period: int = 1):
        self.period = period
        self._coefs: np.ndarray | None = None
        self._n: int = 0
        self._with_exog: bool = False

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> "LinearTrendModel":
        """Fit linear trend (+ seasonal dummies + exog) via OLS."""
        y = np.asarray(y, dtype=float)
        self._n = len(y)
        X = self._build_features(np.arange(self._n), self._n, exog_data=exog)
        self._coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        self._with_exog = exog is not None
        self._X_train = X
        self._y_train = y
        return self

    def predict(
        self,
        steps: int,
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> pd.Series:
        """Produce forecasts for *steps* periods ahead."""
        t = np.arange(self._n, self._n + steps)
        X = self._build_features(t, self._n, exog_data=exog)
        return pd.Series(X @ self._coefs)

    @property
    def aic(self) -> float:
        """Approximate AIC (2k - 2*log-likelihood)."""
        if self._coefs is None:
            return float("inf")
        resid = self._y_train - (self._X_train @ self._coefs)
        k = len(self._coefs)
        n = len(resid)
        sigma2 = np.var(resid)
        if sigma2 <= 0:
            return float("inf")
        loglik = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum(resid ** 2) / (2 * sigma2)
        return 2 * k - 2 * loglik

    @property
    def in_sample_predictions(self) -> pd.Series:
        return pd.Series(self._X_train @ self._coefs)

    @property
    def residuals(self) -> pd.Series:
        return pd.Series(self._y_train - (self._X_train @ self._coefs))

    def _build_features(
        self,
        t: np.ndarray,
        n_train: int,
        exog_data=None,
    ) -> np.ndarray:
        """Construct design matrix: intercept + time + seasonal dummies + exog."""
        parts = [np.ones(len(t)), t / n_train]  # intercept + scaled trend
        if self.period > 1:
            for s in range(self.period - 1):
                dummy = ((t % self.period) == s).astype(float)
                parts.append(dummy)
        if exog_data is not None:
            ex = np.asarray(exog_data)
            if ex.ndim == 1:
                ex = ex[:, None]
            for col in range(ex.shape[1]):
                parts.append(ex[:, col])
        return np.column_stack(parts)
