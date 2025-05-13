"""Tests for metrics calculation functions."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from border_predictor.utils.metrics import calculate_metrics

@pytest.fixture
def sample_data():
    """Create sample actual and forecast data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='H')
    actual = pd.DataFrame({
        'ds': dates,
        'y': [10, 20, 30, 40, 50]
    })
    forecast = pd.DataFrame({
        'ds': dates,
        'yhat': [12, 18, 33, 35, 55]
    })
    return actual, forecast

def test_calculate_metrics(sample_data):
    """Test basic metrics calculation."""
    actual, forecast = sample_data
    metrics = calculate_metrics(actual, forecast)
    
    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ['mae', 'rmse', 'mape'])
    assert metrics['mae'] > 0
    assert metrics['rmse'] > 0
    assert metrics['mape'] > 0

def test_calculate_metrics_perfect_prediction():
    """Test metrics calculation with perfect predictions."""
    dates = pd.date_range(start='2024-01-01', periods=3, freq='H')
    actual = pd.DataFrame({
        'ds': dates,
        'y': [10, 20, 30]
    })
    forecast = pd.DataFrame({
        'ds': dates,
        'yhat': [10, 20, 30]
    })
    
    metrics = calculate_metrics(actual, forecast)
    assert metrics['mae'] == 0
    assert metrics['rmse'] == 0
    assert metrics['mape'] == 0

def test_calculate_metrics_with_zeros():
    """Test metrics calculation with zero values."""
    dates = pd.date_range(start='2024-01-01', periods=3, freq='H')
    actual = pd.DataFrame({
        'ds': dates,
        'y': [0, 20, 30]
    })
    forecast = pd.DataFrame({
        'ds': dates,
        'yhat': [0, 18, 33]
    })
    
    metrics = calculate_metrics(actual, forecast)
    assert not np.isnan(metrics['mae'])
    assert not np.isnan(metrics['rmse'])
    assert not np.isnan(metrics['mape'])

def test_calculate_metrics_empty_data():
    """Test metrics calculation with empty data."""
    actual = pd.DataFrame(columns=['ds', 'y'])
    forecast = pd.DataFrame(columns=['ds', 'yhat'])
    
    metrics = calculate_metrics(actual, forecast)
    assert np.isnan(metrics['mae'])
    assert np.isnan(metrics['rmse'])
    assert np.isnan(metrics['mape'])
