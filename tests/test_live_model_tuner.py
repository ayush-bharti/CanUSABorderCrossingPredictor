"""Tests for the LiveModelTuner class."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from prophet import Prophet
from unittest.mock import patch, MagicMock

from border_predictor.models.live_model_tuner import LiveModelTuner

@pytest.fixture
def sample_model():
    """Create a sample Prophet model for testing."""
    model = Prophet()
    df = pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'y': np.random.normal(30, 10, 100)  # Random wait times around 30 minutes
    })
    model.fit(df)
    return model

@pytest.fixture
def sample_live_data():
    """Create sample live data for testing."""
    return pd.DataFrame({
        'ds': pd.date_range(start='2024-01-01', periods=50, freq='15min'),
        'y': np.random.normal(30, 10, 50)
    })

@pytest.fixture
def temp_live_data_path(tmp_path):
    """Create a temporary path for live data."""
    return tmp_path / "live_data.csv"

def test_init(sample_model, temp_live_data_path):
    """Test LiveModelTuner initialization."""
    tuner = LiveModelTuner(
        model=sample_model,
        live_data_path=temp_live_data_path
    )
    assert isinstance(tuner.model, Prophet)
    assert tuner.collection_interval == 15
    assert tuner.retraining_threshold == 96
    assert isinstance(tuner.live_data, pd.DataFrame)

def test_load_live_data(sample_live_data, temp_live_data_path):
    """Test loading existing live data."""
    sample_live_data.to_csv(temp_live_data_path, index=False)
    tuner = LiveModelTuner(
        model=Prophet(),
        live_data_path=temp_live_data_path
    )
    assert len(tuner.live_data) == len(sample_live_data)
    assert all(tuner.live_data.columns == ['ds', 'y'])

@patch('border_predictor.models.live_model_tuner.get_current_wait_time')
def test_collect_live_data(mock_get_wait_time, sample_model, temp_live_data_path):
    """Test collecting live data."""
    mock_get_wait_time.return_value = (30, {'status': 'open'})
    
    tuner = LiveModelTuner(
        model=sample_model,
        live_data_path=temp_live_data_path
    )
    initial_len = len(tuner.live_data)
    
    wait_time, info = tuner.collect_live_data()
    assert wait_time == 30
    assert info['status'] == 'open'
    assert len(tuner.live_data) == initial_len + 1

def test_should_retrain(sample_model, sample_live_data, temp_live_data_path):
    """Test retraining decision logic."""
    sample_live_data.to_csv(temp_live_data_path, index=False)
    tuner = LiveModelTuner(
        model=sample_model,
        live_data_path=temp_live_data_path,
        retraining_threshold=20
    )
    
    # Should retrain when no previous retraining
    assert tuner.should_retrain() == True
    
    # Should not retrain right after retraining
    tuner.model.last_retrain_time = datetime.now()
    assert tuner.should_retrain() == False

def test_fine_tune_model(sample_model, sample_live_data, temp_live_data_path):
    """Test model fine-tuning."""
    sample_live_data.to_csv(temp_live_data_path, index=False)
    tuner = LiveModelTuner(
        model=sample_model,
        live_data_path=temp_live_data_path,
        retraining_threshold=20
    )
    
    metrics, new_model = tuner.fine_tune_model()
    assert isinstance(metrics, dict)
    assert isinstance(new_model, Prophet)
    assert hasattr(new_model, 'last_retrain_time')

def test_benchmark_predictions(sample_model, sample_live_data, temp_live_data_path):
    """Test prediction benchmarking."""
    sample_live_data.to_csv(temp_live_data_path, index=False)
    tuner = LiveModelTuner(
        model=sample_model,
        live_data_path=temp_live_data_path
    )
    
    metrics = tuner.benchmark_predictions(horizon_hours=1)
    assert isinstance(metrics, dict)
    assert all(horizon in metrics.keys() for horizon in ['1h'])
    assert all(metric in metrics['1h'].keys() for metric in ['mae', 'rmse', 'mape'])
