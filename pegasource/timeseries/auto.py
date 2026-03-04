"""
AutoForecaster: automatic time-series model selection and forecasting.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd

from .models import SARIMAXModel, LinearTrendModel
from .utils import detect_seasonality, rmse


class AutoForecaster:
    """Automatic time-series forecaster with optional exogenous support.

    Tries SARIMAX with multiple (p,d,q) × seasonality configurations and
    falls back to a linear trend model if statsmodels is unavailable or
    fitting fails. Model selection is done via AIC on in-sample fit.

    Parameters
    ----------
    max_p : int
        Maximum AR order to try.
    max_q : int
        Maximum MA order to try.
    seasonal : bool
        If True, also try seasonal orders.

    Examples
    --------
    >>> import numpy as np
    >>> from pegasource.timeseries import AutoForecaster
    >>> t = np.linspace(0, 4 * np.pi, 48)
    >>> y = np.sin(t) + np.random.default_rng(0).normal(0, 0.1, 48)
    >>> fc = AutoForecaster()
    >>> fc.fit(y)
    AutoForecaster(...)
    >>> pred = fc.predict(steps=12)
    """

    def __init__(
        self,
        max_p: int = 2,
        max_q: int = 2,
        seasonal: bool = True,
    ):
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self._model: SARIMAXModel | LinearTrendModel | None = None
        self._period: int = 1
        self._n: int = 0
        self._y: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, list],
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
        freq: str | None = None,
    ) -> "AutoForecaster":
        """Fit the best model to *y*.

        Parameters
        ----------
        y : array-like
            1D time series (endogenous variable).
        exog : 2-D array-like or None
            Exogenous regressors, shape ``(len(y), k)``.
        freq : str or None
            Unused; reserved for future use.

        Returns
        -------
        self
        """
        self._y = np.asarray(y, dtype=float)
        self._n = len(self._y)
        self._period = detect_seasonality(self._y)

        best_aic = float("inf")
        best_model: SARIMAXModel | LinearTrendModel | None = None

        # ---------------- Try SARIMAX candidates ----------------
        candidates = self._sarimax_candidates()
        for order, seasonal_order in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = SARIMAXModel(order=order, seasonal_order=seasonal_order)
                    m.fit(self._y, exog=exog)
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_model = m
            except Exception:
                continue

        # ---------------- Fallback: linear trend ----------------
        lin = LinearTrendModel(period=self._period)
        lin.fit(self._y, exog=exog)
        if lin.aic < best_aic or best_model is None:
            best_model = lin

        self._model = best_model
        self._exog = exog
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        steps: int,
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
    ) -> pd.Series:
        """Produce *steps* out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        exog : array-like or None
            Exogenous regressors for the forecast horizon, shape
            ``(steps, k)``. Required when the model was fitted with exog.

        Returns
        -------
        pd.Series
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        result = self._model.predict(steps=steps, exog=exog)
        # Ensure we always return a pd.Series
        if not isinstance(result, pd.Series):
            result = pd.Series(result)
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Return in-sample diagnostics.

        Returns
        -------
        dict
            Keys: ``model_type``, ``aic``, ``period``, ``in_sample_rmse``,
            ``residuals_mean``, ``residuals_std``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        preds = self._model.in_sample_predictions
        preds_arr = preds.values if hasattr(preds, "values") else np.asarray(preds)
        resid = self._model.residuals
        resid_arr = resid.values if hasattr(resid, "values") else np.asarray(resid)
        return {
            "model_type": type(self._model).__name__,
            "aic": round(self._model.aic, 4),
            "period": self._period,
            "in_sample_rmse": round(rmse(self._y, preds_arr), 6),
            "residuals_mean": round(float(resid_arr.mean()), 6),
            "residuals_std": round(float(resid_arr.std()), 6),
        }

    def plot(
        self,
        steps: int = 12,
        exog: Union[np.ndarray, pd.DataFrame, None] = None,
        title: str = "AutoForecaster",
    ):
        """Plot actuals + in-sample fit + forecast with confidence interval.

        Parameters
        ----------
        steps : int
            Forecast horizon to display.
        exog : array-like or None
            Exogenous values for forecast horizon.
        title : str
            Plot title.
        """
        import matplotlib.pyplot as plt

        if self._model is None:
            raise RuntimeError("Call fit() before plot().")

        fitted_raw = self._model.in_sample_predictions
        fitted = fitted_raw.values if hasattr(fitted_raw, "values") else np.asarray(fitted_raw)
        fc_raw = self._model.predict(steps=steps, exog=exog)
        forecast = fc_raw.values if hasattr(fc_raw, "values") else np.asarray(fc_raw)
        resid_raw = self._model.residuals
        resid_std = float(np.asarray(resid_raw.values if hasattr(resid_raw, "values") else resid_raw).std())

        t_hist = np.arange(self._n)
        t_fore = np.arange(self._n, self._n + steps)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t_hist, self._y, color="#4fc3f7", lw=1.5, label="Actual")
        ax.plot(t_hist, fitted, color="#ff9800", lw=1.5, linestyle="--", label="In-sample fit")
        ax.plot(t_fore, forecast, color="#66bb6a", lw=2, label="Forecast")
        ax.fill_between(
            t_fore,
            forecast - 1.96 * resid_std,
            forecast + 1.96 * resid_std,
            alpha=0.25,
            color="#66bb6a",
            label="95% CI",
        )
        ax.axvline(self._n - 0.5, color="gray", linestyle=":", lw=1)
        ax.set_title(f"{title}  [{type(self._model).__name__}, period={self._period}]")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        model_name = type(self._model).__name__ if self._model else "unfitted"
        return (
            f"AutoForecaster(max_p={self.max_p}, max_q={self.max_q}, "
            f"seasonal={self.seasonal}, fitted_model={model_name})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sarimax_candidates(self):
        """Generate (order, seasonal_order) candidates to try."""
        d = 1  # single differencing
        p_range = range(self.max_p + 1)
        q_range = range(self.max_q + 1)

        # Non-seasonal candidates
        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                yield (p, d, q), (0, 0, 0, 0)

        # Seasonal candidates
        if self.seasonal and self._period > 1:
            for P in range(2):
                for Q in range(2):
                    if P == 0 and Q == 0:
                        continue
                    yield (1, 1, 1), (P, 1, Q, self._period)
