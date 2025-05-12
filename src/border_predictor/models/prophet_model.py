"""Prophet model implementation for border crossing prediction."""
from typing import Dict, List, Optional, Tuple, Union
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt

from border_predictor.config.config import MODEL_CONFIG, BORDER_CONFIG


class BorderProphetModel:
    """Prophet model for border crossing wait time prediction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Prophet model with configuration.
        
        Args:
            config: Optional dictionary of model configuration parameters.
                   If None, uses the default configuration from MODEL_CONFIG.
        """
        self.config = config or MODEL_CONFIG
        self.model = None
        self.last_forecast = None
        self.metrics = None
    
    def build_model(self) -> Prophet:
        """
        Build and configure the Prophet model.
        
        Returns:
            Configured Prophet model instance.
        """
        # Create Prophet model with configuration
        model = Prophet(
            changepoint_prior_scale=self.config["changepoint_prior_scale"],
            seasonality_prior_scale=self.config["seasonality_prior_scale"],
            holidays_prior_scale=self.config["holidays_prior_scale"],
            daily_seasonality=self.config["daily_seasonality"],
            weekly_seasonality=self.config["weekly_seasonality"],
            yearly_seasonality=self.config["yearly_seasonality"],
        )
        
        # Add country holidays
        for country in BORDER_CONFIG["countries"]:
            model.add_country_holidays(country_name=country)
        
        return model
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the Prophet model to the data.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns.
        """
        # Build the model
        self.model = self.build_model()
        
        # Add any additional regressors here if needed
        # Example: self.model.add_regressor('is_weekend')
        
        # Fit the model
        self.model.fit(df)
        
        print(f"Model fitted on {len(df)} data points.")
    
    def predict(
        self, 
        periods: Optional[int] = None, 
        future_df: Optional[pd.DataFrame] = None,
        include_history: bool = False
    ) -> pd.DataFrame:
        """
        Generate predictions using the fitted model.
        
        Args:
            periods: Number of future periods to predict. Ignored if future_df is provided.
            future_df: Optional DataFrame with future dates for prediction.
            include_history: Whether to include historical predictions.
        
        Returns:
            DataFrame with predictions.
        
        Raises:
            ValueError: If the model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if future_df is None:
            # Create future dataframe
            if periods is None:
                periods = self.config["forecast_horizon_days"] * 24  # Convert days to hours
            
            future_df = self.model.make_future_dataframe(
                periods=periods,
                freq=self.config["forecast_frequency"],
                include_history=include_history
            )
        
        # Make predictions
        forecast = self.model.predict(future_df)
        self.last_forecast = forecast
        
        return forecast
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        initial: Optional[str] = None,
        period: Optional[str] = None,
        horizon: Optional[str] = None
    ) -> Dict:
        """
        Evaluate the model using cross-validation.
        
        Args:
            df: DataFrame with 'ds' and 'y' columns.
            initial: Initial training period, e.g., '30 days'.
            period: Period between cutoffs, e.g., '7 days'.
            horizon: Forecast horizon, e.g., '7 days'.
        
        Returns:
            Dictionary of performance metrics.
        
        Raises:
            ValueError: If the model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Set default values if not provided
        if initial is None:
            initial = '30 days'
        if period is None:
            period = '7 days'
        if horizon is None:
            horizon = f"{self.config['forecast_horizon_days']} days"
        
        # Perform cross-validation
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        # Calculate performance metrics
        metrics = performance_metrics(cv_results)
        self.metrics = metrics
        
        return metrics
    
    def plot_forecast(
        self, 
        forecast: Optional[pd.DataFrame] = None, 
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot the forecast.
        
        Args:
            forecast: Optional forecast DataFrame. If None, uses the last forecast.
            figsize: Figure size as (width, height).
        
        Returns:
            Matplotlib figure.
        
        Raises:
            ValueError: If no forecast is available.
        """
        if forecast is None:
            forecast = self.last_forecast
        
        if forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Plot forecast
        self.model.plot(forecast, fig=fig)
        plt.title(f"Border Crossing Wait Time Forecast - {BORDER_CONFIG['crossing_name']}")
        plt.xlabel("Date")
        plt.ylabel("Wait Time (minutes)")
        
        return fig
    
    def plot_components(
        self, 
        forecast: Optional[pd.DataFrame] = None, 
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot the forecast components.
        
        Args:
            forecast: Optional forecast DataFrame. If None, uses the last forecast.
            figsize: Figure size as (width, height).
        
        Returns:
            Matplotlib figure.
        
        Raises:
            ValueError: If no forecast is available.
        """
        if forecast is None:
            forecast = self.last_forecast
        
        if forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Plot components
        self.model.plot_components(forecast, fig=fig)
        
        return fig
    
    def save_model(self, file_path: Optional[Path] = None) -> None:
        """
        Save the model to a file.
        
        Args:
            file_path: Path to save the model. If None, uses the default path
                      from MODEL_CONFIG.
        
        Raises:
            ValueError: If the model has not been fitted.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if file_path is None:
            file_path = self.config["model_path"]
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: Optional[Path] = None) -> None:
        """
        Load the model from a file.
        
        Args:
            file_path: Path to load the model from. If None, uses the default path
                      from MODEL_CONFIG.
        
        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        if file_path is None:
            file_path = self.config["model_path"]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at {file_path}")
        
        # Load model
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {file_path}")
    
    def get_forecast_for_date(
        self, 
        target_date: Union[str, pd.Timestamp], 
        forecast: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get forecast for a specific date.
        
        Args:
            target_date: Target date to get forecast for.
            forecast: Optional forecast DataFrame. If None, uses the last forecast.
        
        Returns:
            DataFrame with forecast for the target date.
        
        Raises:
            ValueError: If no forecast is available.
        """
        if forecast is None:
            forecast = self.last_forecast
        
        if forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        # Convert target_date to pandas Timestamp
        if isinstance(target_date, str):
            target_date = pd.Timestamp(target_date)
        
        # Filter forecast for the target date
        target_forecast = forecast[forecast['ds'].dt.date == target_date.date()]
        
        return target_forecast
