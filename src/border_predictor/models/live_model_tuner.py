"""Module for continuous model fine-tuning using live border crossing data."""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from prophet import Prophet

from border_predictor.data.live_data_loader import get_current_wait_time, prepare_live_data_for_model, save_live_data
from border_predictor.config.config import MODEL_CONFIG, DATA_CONFIG
from border_predictor.utils.metrics import calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveModelTuner:
    """Class for continuous model fine-tuning using live border crossing data."""
    
    def __init__(
        self,
        model: Prophet,
        live_data_path: Optional[Path] = None,
        collection_interval: int = 15,  # minutes
        retraining_threshold: int = 96,  # number of new data points before retraining
        max_retrain_window: int = 336  # hours of historical data to use for retraining
    ):
        """
        Initialize the LiveModelTuner.
        
        Args:
            model: Trained Prophet model to fine-tune
            live_data_path: Path to store live data
            collection_interval: Minutes between data collection
            retraining_threshold: Number of new points before retraining
            max_retrain_window: Hours of historical data to use for retraining
        """
        self.model = model
        self.live_data_path = live_data_path or DATA_CONFIG["live_data_path"]
        self.collection_interval = collection_interval
        self.retraining_threshold = retraining_threshold
        self.max_retrain_window = max_retrain_window
        self.live_data = self._load_live_data()
        
    def _load_live_data(self) -> pd.DataFrame:
        """Load existing live data or create empty DataFrame."""
        try:
            df = pd.read_csv(self.live_data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=['ds', 'y'])
    
    def collect_live_data(self) -> Tuple[int, Dict]:
        """
        Collect current wait time data.
        
        Returns:
            Tuple containing:
            - Current wait time in minutes
            - Dictionary with additional crossing information
        """
        try:
            # Get current time for logging
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Starting data collection at {current_time}")
            
            # Get wait time from API
            wait_time, info = get_current_wait_time()
            logger.info(f"API response: wait_time={wait_time}, crossing_info={info}")
            
            # Prepare data for model
            new_data = prepare_live_data_for_model(wait_time)
            logger.info(
                f"Prepared data point: timestamp={new_data['ds'].iloc[0]} "
                f"wait_time={new_data['y'].iloc[0]} "
                f"hour={new_data['hour'].iloc[0]} "
                f"day={new_data['day'].iloc[0]} "
                f"month={new_data['month'].iloc[0]} "
                f"year={new_data['year'].iloc[0]}"
            )
            
            # Append new data
            prev_len = len(self.live_data)
            self.live_data = pd.concat([self.live_data, new_data], ignore_index=True)
            logger.info(f"Added to live data collection (total: {len(self.live_data)} records)")
            
            # Save to file
            save_live_data(new_data, self.live_data_path)
            
            # Summary log
            logger.info(f"Collected new wait time: {wait_time} minutes")
            return wait_time, info
            
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained based on new data points."""
        if len(self.live_data) < self.retraining_threshold:
            return False
            
        last_retrain_time = getattr(self.model, 'last_retrain_time', None)
        if last_retrain_time is None:
            return True
            
        new_points_since_retrain = len(
            self.live_data[self.live_data['ds'] > last_retrain_time]
        )
        return new_points_since_retrain >= self.retraining_threshold
    
    def fine_tune_model(self) -> Tuple[Dict[str, float], Prophet]:
        """
        Fine-tune the model with recent live data.
        
        Returns:
            Tuple containing:
            - Dictionary with performance metrics
            - Fine-tuned Prophet model
        """
        # Get training data within max_retrain_window
        cutoff_time = datetime.now() - timedelta(hours=self.max_retrain_window)
        train_df = self.live_data[self.live_data['ds'] > cutoff_time].copy()
        
        if len(train_df) < self.retraining_threshold:
            logger.warning("Insufficient data for retraining")
            # Even when not retraining, set the last_retrain_time to track when we checked
            setattr(self.model, 'last_retrain_time', datetime.now())
            return {}, self.model
            
        try:
            # Create new model with same parameters from inner Prophet model
            prophet_model = self.model.model
            new_model = Prophet(
                changepoint_prior_scale=prophet_model.changepoint_prior_scale,
                seasonality_prior_scale=prophet_model.seasonality_prior_scale,
                holidays_prior_scale=prophet_model.holidays_prior_scale,
                seasonality_mode=prophet_model.seasonality_mode
            )
            
            # Add any custom seasonalities from original model
            for s in self.model.seasonalities:
                if s not in ['daily', 'weekly', 'yearly']:
                    params = self.model.seasonalities[s]
                    new_model.add_seasonality(
                        name=s,
                        period=params['period'],
                        fourier_order=params['fourier_order']
                    )
            
            # Fit model
            new_model.fit(train_df)
            
            # Calculate performance metrics
            future = new_model.make_future_dataframe(periods=24, freq='H')
            forecast = new_model.predict(future)
            metrics = calculate_metrics(train_df, forecast)
            
            logger.info("Model successfully fine-tuned")
            logger.info(f"New metrics: {metrics}")
            
            # Set last_retrain_time after successful fine-tuning
            setattr(new_model, 'last_retrain_time', datetime.now())
            
            return metrics, new_model
            
        except Exception as e:
            logger.error(f"Error during model fine-tuning: {e}")
            raise
    
    def benchmark_predictions(
        self,
        horizon_hours: int = 24
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark model predictions against actual values.
        
        Args:
            horizon_hours: Number of hours to look back for benchmarking
            
        Returns:
            Dictionary with performance metrics for different prediction horizons
        """
        cutoff_time = datetime.now() - timedelta(hours=horizon_hours)
        actual_data = self.live_data[self.live_data['ds'] > cutoff_time].copy()
        
        if len(actual_data) < 2:
            logger.warning("Insufficient data for benchmarking")
            return {'1h': {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}}
            
        # Make predictions for the past horizon_hours
        future = pd.DataFrame({'ds': actual_data['ds']})
        forecast = self.model.predict(future)
        
        # Calculate metrics for different time horizons
        horizons = {
            '1h': 1,
            '4h': 4,
            '12h': 12,
            '24h': 24
        }
        
        metrics = {}
        for label, hours in horizons.items():
            if hours <= horizon_hours:
                horizon_data = actual_data.tail(hours)
                horizon_forecast = forecast.tail(hours)
                metrics[label] = calculate_metrics(horizon_data, horizon_forecast)
                
        return metrics
