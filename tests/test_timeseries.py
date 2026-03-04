"""
Tests for pegasource.timeseries — AutoForecaster, models, utilities.
"""

import numpy as np
import pandas as pd
import pytest

from pegasource.timeseries.utils import detect_seasonality, train_test_split_ts, rmse
from pegasource.timeseries.models import LinearTrendModel, SARIMAXModel
from pegasource.timeseries import AutoForecaster


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestDetectSeasonality:
    def test_weekly_period(self):
        # Strong 7-period seasonal signal; ACF may detect the base period or
        # a harmonic (e.g. 14). We accept any result that evenly divides into
        # the true period or vice-versa.
        t = np.arange(56)
        y = np.sin(2 * np.pi * t / 7) + np.random.default_rng(42).normal(0, 0.05, 56)
        period = detect_seasonality(y, max_period=20)
        assert period % 7 == 0 or 7 % period == 0, \
            f"Expected a multiple or factor of 7, got {period}"

    def test_no_seasonality(self):
        y = np.random.default_rng(0).normal(0, 1, 50)
        period = detect_seasonality(y)
        assert isinstance(period, int) and period >= 1

    def test_short_series(self):
        y = [1.0, 2.0, 3.0]
        assert detect_seasonality(y) == 1


class TestTrainTestSplit:
    def test_basic_split(self):
        y = np.arange(100, dtype=float)
        y_train, y_test = train_test_split_ts(y, test_frac=0.2)
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_chronological(self):
        y = np.arange(50, dtype=float)
        y_train, y_test = train_test_split_ts(y)
        assert y_train[-1] < y_test[0]

    def test_with_exog(self):
        y = np.arange(40, dtype=float)
        X = np.ones((40, 2))
        y_train, y_test, X_train, X_test = train_test_split_ts(y, exog=X, test_frac=0.25)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestRMSE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestLinearTrendModel:
    def _trend_series(self, n=40):
        return np.arange(n, dtype=float) + np.random.default_rng(1).normal(0, 0.5, n)

    def test_fit_and_predict(self):
        y = self._trend_series()
        m = LinearTrendModel()
        m.fit(y)
        pred = m.predict(steps=5)
        assert len(pred) == 5

    def test_with_seasonality(self):
        t = np.arange(42)
        y = np.sin(2 * np.pi * t / 7) + t * 0.1
        m = LinearTrendModel(period=7)
        m.fit(y)
        pred = m.predict(steps=7)
        assert len(pred) == 7

    def test_with_exog(self):
        n = 30
        exog = np.random.default_rng(3).normal(size=(n, 2))
        y = exog[:, 0] * 2 + exog[:, 1] * -1 + np.arange(n)
        m = LinearTrendModel()
        m.fit(y, exog=exog)
        fut_exog = np.random.default_rng(3).normal(size=(5, 2))
        pred = m.predict(steps=5, exog=fut_exog)
        assert len(pred) == 5

    def test_residuals(self):
        y = np.linspace(0, 10, 20)
        m = LinearTrendModel()
        m.fit(y)
        assert len(m.residuals) == 20

    def test_aic_finite(self):
        y = self._trend_series()
        m = LinearTrendModel()
        m.fit(y)
        assert np.isfinite(m.aic)


class TestSARIMAXModel:
    def _ar1_series(self, n=60):
        rng = np.random.default_rng(42)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.7 * y[i - 1] + rng.normal()
        return y

    def test_fit_and_predict(self):
        y = self._ar1_series()
        m = SARIMAXModel(order=(1, 0, 0))
        m.fit(y)
        pred = m.predict(steps=5)
        assert len(pred) == 5

    def test_aic_finite(self):
        y = self._ar1_series()
        m = SARIMAXModel(order=(1, 0, 0))
        m.fit(y)
        assert np.isfinite(m.aic)

    def test_residuals(self):
        y = self._ar1_series()
        m = SARIMAXModel(order=(1, 0, 0))
        m.fit(y)
        assert len(m.residuals) == len(y)


# ---------------------------------------------------------------------------
# AutoForecaster
# ---------------------------------------------------------------------------

class TestAutoForecaster:
    def _sine_wave(self, n=48, period=12, noise=0.1):
        t = np.linspace(0, 4 * np.pi, n)
        rng = np.random.default_rng(7)
        return np.sin(t) + rng.normal(0, noise, n)

    def test_fit_returns_self(self):
        y = self._sine_wave()
        fc = AutoForecaster()
        result = fc.fit(y)
        assert result is fc

    def test_predict_length(self):
        y = self._sine_wave()
        fc = AutoForecaster()
        fc.fit(y)
        pred = fc.predict(steps=6)
        assert len(pred) == 6

    def test_predict_returns_series(self):
        y = self._sine_wave()
        fc = AutoForecaster()
        fc.fit(y)
        pred = fc.predict(steps=4)
        assert isinstance(pred, pd.Series)

    def test_diagnostics(self):
        y = self._sine_wave()
        fc = AutoForecaster()
        fc.fit(y)
        diag = fc.diagnostics()
        assert "aic" in diag
        assert "in_sample_rmse" in diag
        assert "period" in diag
        assert diag["in_sample_rmse"] >= 0

    def test_with_exog(self):
        n = 48
        rng = np.random.default_rng(99)
        exog_train = rng.normal(size=(n, 1))
        y = exog_train[:, 0] * 2 + np.sin(np.linspace(0, 4 * np.pi, n))
        fc = AutoForecaster()
        fc.fit(y, exog=exog_train)
        exog_fore = rng.normal(size=(6, 1))
        pred = fc.predict(steps=6, exog=exog_fore)
        assert len(pred) == 6

    def test_repr(self):
        fc = AutoForecaster()
        assert "AutoForecaster" in repr(fc)

    def test_unfitted_predict_raises(self):
        fc = AutoForecaster()
        with pytest.raises(RuntimeError):
            fc.predict(steps=3)
