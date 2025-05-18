"""Module for continuous model fine-tuning using live border crossing data."""
import sys
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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('live_model_tuner.log')
    ]
)
logger = logging.getLogger(__name__)

# Add a console handler for better visibility
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

class LiveModelTuner:
    """Class for continuous model fine-tuning using live border crossing data."""
    
    PERFORMANCE_THRESHOLD = {
        'mae': 10,  # minutes
        'rmse': 15,  # minutes
        'mape': 25.0  # 25%
    }
    
    def __init__(
        self,
        model,  # Can be either Prophet or BorderProphetModel
        live_data_path: Optional[Path] = None,
        collection_interval: int = 15,  # minutes
        retraining_threshold: int = 96,  # number of new data points before retraining
        max_retrain_window: int = 336  # hours of historical data to use for retraining
    ):
        """
        Initialize the LiveModelTuner.
        
        Args:
            model: Either a Prophet model or a BorderProphetModel instance to fine-tune
            live_data_path: Path to store live data
            collection_interval: Minutes between data collection
            retraining_threshold: Number of new points before retraining
            max_retrain_window: Hours of historical data to use for retraining
        """
        # Store model and identify its type
        self.model = model
        self.is_border_model = hasattr(model, 'model')  # True if BorderProphetModel, False if Prophet
        
        # Log model type for debugging
        logger.info(f"Initialized with model type: {type(model).__name__}")
        if self.is_border_model:
            logger.info(f"Inner model exists: {model.model is not None}")
            if model.model is not None:
                logger.info(f"Inner model type: {type(model.model).__name__}")
        
        # Initialize other settings
        self.live_data_path = live_data_path or DATA_CONFIG["live_data_path"]
        self.collection_interval = collection_interval
        self.retraining_threshold = retraining_threshold
        self.max_retrain_window = max_retrain_window
        self.live_data = self._load_live_data()
        
        # Ensure model is properly initialized
        self._validate_model()
        
    def _validate_model(self):
        """Validate and ensure the model is properly initialized."""
        if self.is_border_model:
            # We have a BorderProphetModel instance
            if not hasattr(self.model, 'model') or self.model.model is None:
                logger.warning("BorderProphetModel does not have an initialized inner Prophet model")
                # We don't raise an error here as we'll handle this during fine-tuning
        else:
            # We have a direct Prophet model instance
            if not isinstance(self.model, Prophet):
                logger.warning(f"Expected Prophet model, got {type(self.model).__name__}")
        
    def _load_live_data(self) -> pd.DataFrame:
        """Load existing live data or create empty DataFrame."""
        try:
            if self.live_data_path.exists():
                logger.info(f"Attempting to load live data from {self.live_data_path}")
                df = pd.read_csv(self.live_data_path, parse_dates=['ds'])
                
                # Log basic statistics about the loaded data
                if not df.empty:
                    logger.info(f"Successfully loaded {len(df)} data points")
                    logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
                    logger.info(f"Wait time stats - Min: {df['y'].min():.1f}, "
                                f"Max: {df['y'].max():.1f}, "
                                f"Mean: {df['y'].mean():.1f} minutes")
                else:
                    logger.warning("Loaded an empty DataFrame from existing file")
                    
                return df
            else:
                logger.info(f"No existing live data found at {self.live_data_path}. Starting with an empty dataset.")
                return pd.DataFrame(columns=['ds', 'y', 'hour', 'day_of_week', 'month', 'day', 'year', 'is_weekend'])
                
        except Exception as e:
            logger.error(f"Error loading live data from {self.live_data_path}")
            logger.error(f"Error details: {str(e)}")
            logger.error("Creating new empty DataFrame")
            logger.debug("Stack trace:", exc_info=True)
            return pd.DataFrame(columns=['ds', 'y', 'hour', 'day_of_week', 'month', 'day', 'year', 'is_weekend'])
    
    def collect_live_data(self) -> Tuple[Optional[int], Dict]:
        """
        Collect current wait time data.
        
        Returns:
            Tuple containing:
            - Current wait time in minutes (or None if failed)
            - Dictionary with additional crossing information
        """
        logger.info("Starting data collection...")
        try:
            # Get current wait time from API
            logger.debug("Fetching current wait time from API...")
            wait_time, info = get_current_wait_time()
            
            if wait_time is None:
                logger.warning("Received None as wait time from API")
                return None, info or {}
                
            logger.info(f"Current wait time: {wait_time} minutes")
            logger.debug(f"API response: {info}")
            
            # Prepare data for model
            logger.debug("Preparing data for model...")
            new_data = prepare_live_data_for_model(wait_time, info)
            
            if new_data is None or new_data.empty:
                logger.error("Failed to prepare data for model - empty or None data returned")
                return wait_time, info or {}
                
            # Log the prepared data
            logger.debug(f"Prepared data: {new_data.to_dict(orient='records')[0]}")
            
            # Append new data
            prev_len = len(self.live_data)
            self.live_data = pd.concat([self.live_data, new_data], ignore_index=True)
            logger.info(f"Added to live data collection (total: {len(self.live_data)} records, added: {len(self.live_data) - prev_len})")
            
            # Save to file
            logger.debug(f"Saving updated data to {self.live_data_path}")
            save_live_data(self.live_data, self.live_data_path)
            
            # Log data statistics
            if not self.live_data.empty:
                logger.debug(f"Current data range: {self.live_data['ds'].min()} to {self.live_data['ds'].max()}")
                logger.debug(f"Wait time stats - Min: {self.live_data['y'].min()}, "
                            f"Max: {self.live_data['y'].max()}, "
                            f"Mean: {self.live_data['y'].mean():.1f} minutes")
            
            logger.info(f"Successfully collected and saved wait time: {wait_time} minutes")
            return wait_time, info
            
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            logger.error(f"Error details: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            raise
    
    def should_retrain(self) -> bool:
        """
        Check if model should be retrained based on new data points.
        
        Returns:
            bool: True if model should be retrained, False otherwise
        """
        logger.debug("Checking if model should be retrained...")
        
        if self.live_data.empty:
            logger.warning("No data available for retraining check")
            return False
            
        # Check if we have enough data points
        data_points = len(self.live_data)
        if data_points < self.retraining_threshold:
            logger.info(f"Not enough data points for retraining. Have {data_points}, need {self.retraining_threshold}")
            return False
            
        logger.debug(f"Sufficient data points available: {data_points} (threshold: {self.retraining_threshold})")
        
        # Log information about the most recent data point
        last_data_point = pd.to_datetime(self.live_data['ds'].max())
        time_since_last_point = datetime.now() - last_data_point
        
        if time_since_last_point < timedelta(minutes=self.collection_interval * 2):
            logger.debug(f"Recent data point from {last_data_point} ({time_since_last_point} ago)")
        else:
            logger.warning(f"Last data point is old: {last_data_point} ({time_since_last_point} ago)")
        
        # Always return True if we have enough data points, regardless of when we last retrained
        logger.info(
            f"Retraining conditions met: "
            f"{data_points} data points (≥{self.retraining_threshold})"
        )
        return True
    
    def fine_tune_model(self) -> Tuple[Dict[str, float], object]:
        """
        Fine-tune the model with recent live data.
        
        Returns:
            Tuple containing:
            - Dictionary with performance metrics
            - Fine-tuned model (either BorderProphetModel or Prophet)
        """
        logger.info("Starting model fine-tuning...")
        start_time = datetime.now()
        
        try:
            # Log data statistics before fine-tuning
            logger.info(f"Fine-tuning with {len(self.live_data)} data points")
            if not self.live_data.empty:
                logger.debug(f"Data range: {self.live_data['ds'].min()} to {self.live_data['ds'].max()}")
                logger.debug(f"Wait time stats - Min: {self.live_data['y'].min()}, "
                            f"Max: {self.live_data['y'].max()}, "
                            f"Mean: {self.live_data['y'].mean():.1f} minutes")
            
            # Prepare data for fine-tuning
            logger.debug("Preparing training data...")
            cutoff_time = datetime.now() - timedelta(hours=self.max_retrain_window)
            train_df = self.live_data[self.live_data['ds'] > cutoff_time].copy()
            
            if len(train_df) < self.retraining_threshold:
                logger.warning(
                    f"Insufficient data for retraining. Have {len(train_df)}, "
                    f"need {self.retraining_threshold}"
                )
                return {}, self.model
                
            logger.info(f"Training on {len(train_df)} data points")
            
            # Get the appropriate model instance
            if self.is_border_model:
                if not hasattr(self.model, 'model') or self.model.model is None:
                    logger.error("BorderProphetModel doesn't have an initialized inner Prophet model")
                    return {}, self.model
                prophet_model = self.model.model
                logger.debug(f"Using inner Prophet model from BorderProphetModel")
            else:
                prophet_model = self.model
                logger.debug("Using direct Prophet model")
            
            # Create new Prophet model with same parameters
            logger.info("Creating new Prophet model with extracted parameters")
            try:
                new_prophet = Prophet(
                    changepoint_prior_scale=prophet_model.changepoint_prior_scale,
                    seasonality_prior_scale=prophet_model.seasonality_prior_scale,
                    holidays_prior_scale=prophet_model.holidays_prior_scale,
                    seasonality_mode=prophet_model.seasonality_mode,
                    growth='logistic'  # Maintain same growth settings as original model
                )
                logger.debug("Successfully created Prophet model with extracted parameters")
            except Exception as e:
                logger.error(f"Error creating Prophet model with parameters: {e}")
                logger.info("Falling back to default parameters")
                new_prophet = Prophet(growth='logistic')
                
            # Add any custom seasonalities from original model
            if hasattr(prophet_model, 'seasonalities'):
                logger.debug("Adding custom seasonalities from original model...")
                for s in prophet_model.seasonalities:
                    if s not in ['daily', 'weekly', 'yearly']:
                        params = prophet_model.seasonalities[s]
                        new_prophet.add_seasonality(
                            name=s,
                            period=params['period'],
                            fourier_order=params['fourier_order']
                        )
                        logger.debug(f"Added custom seasonality: {s} (period={params['period']}, "
                                    f"fourier_order={params['fourier_order']})")
            
            # Prepare data for logistic growth
            max_wait_time = train_df['y'].max()
            cap_value = max(120, max_wait_time * 2)  # Use at least 120 mins or 2x max value
            train_df['cap'] = cap_value
            train_df['floor'] = 0  # Wait times can't go below 0
            
            logger.info(f"Fitting Prophet model on {len(train_df)} data points with cap={cap_value}")
            logger.debug(f"Train data columns: {train_df.columns.tolist()}")
            
            # Fit the model
            fit_start = datetime.now()
            new_prophet.fit(train_df)
            fit_duration = (datetime.now() - fit_start).total_seconds()
            logger.info(f"Model fitting completed in {fit_duration:.2f} seconds")
            
            # Calculate performance metrics
            logger.debug("Calculating performance metrics...")
            future = new_prophet.make_future_dataframe(periods=24, freq='H')
            future['cap'] = cap_value
            future['floor'] = 0
            forecast = new_prophet.predict(future)
            metrics = calculate_metrics(train_df, forecast)
            
            # Log performance metrics
            logger.info("Model performance metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.2f}")
                if metric in self.PERFORMANCE_THRESHOLD and value > self.PERFORMANCE_THRESHOLD[metric]:
                    logger.warning(
                        f"{metric.upper()} ({value:.2f}) exceeds threshold "
                        f"({self.PERFORMANCE_THRESHOLD[metric]})"
                    )
            
            # Record the last_retrain_time after successful fine-tuning
            current_time = datetime.now()
            total_duration = (current_time - start_time).total_seconds()
            logger.info(f"Model fine-tuning completed in {total_duration:.2f} seconds")
            
            # Return the appropriate model type to match what was provided
            if self.is_border_model:
                from border_predictor.models.prophet_model import BorderProphetModel
                
                logger.debug("Creating new BorderProphetModel wrapper...")
                new_border_model = BorderProphetModel(config=getattr(self.model, 'config', {}))
                new_border_model.model = new_prophet
                new_border_model.metrics = metrics
                setattr(new_border_model, 'last_retrain_time', current_time)
                
                logger.info("Successfully created new BorderProphetModel with fine-tuned Prophet model")
                return metrics, new_border_model
            else:
                setattr(new_prophet, 'last_retrain_time', current_time)
                setattr(new_prophet, 'metrics', metrics)
                logger.info("Successfully created new Prophet model")
                return metrics, new_prophet
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error during model fine-tuning after {duration:.2f} seconds: {str(e)}")
            logger.debug("Stack trace:", exc_info=True)
            return {}, self.model
    
    def run_fine_tuning_cycle(self) -> Tuple[bool, Dict[str, float]]:
        """
        Run a complete fine-tuning cycle including data collection, model update,
        and performance validation.
        
        Returns:
            Tuple containing:
            - Boolean indicating if fine-tuning was successful
            - Dictionary with performance metrics
        """
        try:
            # Check if we should retrain
            if not self.should_retrain():
                logger.info("Skipping fine-tuning - insufficient new data points")
                return False, {}
            
            # Fine-tune the model
            metrics, new_model = self.fine_tune_model()
            
            if not metrics:
                logger.info("No metrics returned - fine-tuning skipped")
                return False, {}
            
            # Validate performance
            if self._validate_performance(metrics):
                # Update the model if performance is acceptable
                self.model = new_model
                logger.info("Model successfully updated with fine-tuned version")
                return True, metrics
            else:
                logger.warning("Fine-tuned model did not meet performance thresholds")
                return False, metrics
                
        except Exception as e:
            logger.error(f"Error during fine-tuning cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, {}
    
    def _validate_performance(self, metrics: Dict[str, float]) -> bool:
        """
        Validate if the fine-tuned model meets performance thresholds.
        
        Args:
            metrics: Dictionary with model performance metrics
            
        Returns:
            Boolean indicating if performance is acceptable
        """
        for metric, threshold in self.PERFORMANCE_THRESHOLD.items():
            if metric in metrics and metrics[metric] > threshold:
                logger.warning(
                    f"Performance validation failed for {metric}: "
                    f"{metrics[metric]:.2f} > {threshold:.2f}"
                )
                return False
        return True

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


def main():
    """Main function to run the live model tuner."""
    try:
        logger.info("Starting Live Model Tuner")
        
        # Import here to avoid circular imports
        from border_predictor.models.prophet_model import BorderProphetModel
        from border_predictor.config.config import MODEL_CONFIG, DATA_CONFIG
        import pandas as pd
        
        # Initialize model with Prophet model explicitly created
        logger.info("Initializing Prophet model with enhanced responsiveness to recent patterns...")
        prophet_model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_prior_scale=10.0,  # Increase sensitivity to recent patterns
            changepoint_prior_scale=0.5,   # Allow for more flexible trend changes
            changepoint_range=0.95         # Allow changepoints closer to the end of the time series
        )
        
        # Create BorderProphetModel and assign Prophet model to it
        logger.info("Initializing BorderProphetModel...")
        model = BorderProphetModel(config=MODEL_CONFIG)
        model.model = prophet_model
        logger.info(f"Prophet model initialized: {model.model is not None}")
        
        # Load historical data for initial training if needed
        historical_data_path = Path(DATA_CONFIG["historical_data_path"])
        if historical_data_path.exists():
            logger.info(f"Loading historical data from {historical_data_path}")
            try:
                historical_data = pd.read_csv(historical_data_path)
                if not historical_data.empty:
                    logger.info(f"Loaded {len(historical_data)} historical data points")
                    model.fit(historical_data)
                    logger.info("Initial model training completed with historical data")
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
        
        # Create and run the tuner
        logger.info("Creating LiveModelTuner...")
        tuner = LiveModelTuner(
            model=model,
            live_data_path=Path(DATA_CONFIG["live_data_path"]),
            collection_interval=15,  # minutes
            retraining_threshold=96,  # data points
            max_retrain_window=336    # hours (2 weeks)
        )
        
        # Run a single fine-tuning cycle
        logger.info("Running fine-tuning cycle...")
        success, metrics = tuner.run_fine_tuning_cycle()
        
        if success:
            logger.info(f"Fine-tuning completed successfully. Metrics: {metrics}")
        else:
            logger.warning(f"Fine-tuning did not complete successfully. Metrics: {metrics}")
            
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
