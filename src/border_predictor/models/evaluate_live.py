"""Module for evaluating model predictions against live border crossing wait times."""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import os

from border_predictor.config.config import MODEL_CONFIG, DATA_CONFIG, LIVE_DATA_CONFIG, BORDER_CONFIG
from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.data.live_data_loader_updated import get_current_wait_time, prepare_live_data_for_model, save_live_data


def load_model(model_path: Optional[Path] = None) -> BorderProphetModel:
    """
    Load a trained Prophet model.
    
    Args:
        model_path: Path to the model file. If None, uses the path from config.
        
    Returns:
        Loaded model.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if model_path is None:
        model_path = MODEL_CONFIG["model_path"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def evaluate_with_live_data(
    model: Optional[BorderProphetModel] = None,
    forecast_hours: int = 24,
    crossing_name: Optional[str] = None,
    lane_type: Optional[str] = None,
    api_url: Optional[str] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate model predictions against live border crossing wait times.
    
    Args:
        model: Optional trained model. If None, loads the model from the default path.
        forecast_hours: Number of hours to forecast.
        crossing_name: Name of the border crossing. If None, uses default from config.
        lane_type: Type of lane ('standard' or 'nexus'). If None, uses default from config.
        api_url: URL of the CBP API. If None, uses default from config.
        
    Returns:
        Tuple of (evaluation_results_df, mean_absolute_error)
    """
    # Set defaults from config
    if crossing_name is None:
        crossing_name = BORDER_CONFIG["crossing_name"]
    
    if lane_type is None:
        lane_type = BORDER_CONFIG.get("lane_type", "standard")
    
    if api_url is None:
        api_url = LIVE_DATA_CONFIG["api_url"]
    
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Get current wait time
    try:
        port_name = BORDER_CONFIG.get("port_name", "Blaine")
        current_wait_time, crossing_info = get_current_wait_time(
            port_name=port_name,
            crossing_name=crossing_name,
            lane_type=lane_type,
            api_url=api_url
        )
        
        print(f"Current wait time at {crossing_info.get('crossing_name', crossing_name)} ({lane_type} lanes): {current_wait_time} minutes")
        
        # Prepare live data for model
        live_df = prepare_live_data_for_model(current_wait_time)
        
        # Save live data
        save_live_data(live_df)
        
        # Generate future timestamps for prediction
        now = datetime.now()
        future_times = [now + timedelta(hours=i) for i in range(forecast_hours)]
        future_df = pd.DataFrame({"ds": future_times})
        
        # Make predictions
        forecast = model.predict(future_df=future_df, include_history=True)
        
        # Get prediction for current hour
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        predictions = forecast[forecast['ds'] == pd.Timestamp(current_hour)]
        
        # If no prediction for exact current hour, use the hour average
        if len(predictions) > 0:
            current_prediction = predictions['yhat'].values[0]
        else:
            # Fallback to same hour predictions if available
            hour_predictions = forecast[forecast['ds'].dt.hour == now.hour]
            if len(hour_predictions) > 0:
                current_prediction = hour_predictions['yhat'].mean()
            else:
                # Ultimate fallback - use the mean of all predictions
                current_prediction = forecast['yhat'].mean()
        
        # Ensure prediction is non-negative
        if current_prediction < 0:
            current_prediction = 0
        
        # Evaluate the current prediction
        actual = current_wait_time
        predicted = current_prediction
        error = abs(actual - predicted)
        
        print(f"Actual wait time: {actual} minutes")
        print(f"Predicted wait time for current hour: {predicted:.2f} minutes")
        print(f"Absolute error: {error:.2f} minutes")
        
        # Create evaluation results
        eval_df = pd.DataFrame({
            "timestamp": [now],
            "actual_wait_time": [actual],
            "predicted_wait_time": [predicted],
            "absolute_error": [error]
        })
        
        # Save evaluation results
        eval_path = DATA_CONFIG["data_dir"] / "model_evaluation.csv"
        if os.path.exists(eval_path):
            existing_eval = pd.read_csv(eval_path)
            updated_eval = pd.concat([existing_eval, eval_df])
            updated_eval.to_csv(eval_path, index=False)
        else:
            eval_df.to_csv(eval_path, index=False)
        
        return eval_df, error
        
    except Exception as e:
        print(f"Error evaluating with live data: {e}")
        raise


def plot_evaluation_history(
    eval_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot the history of model evaluation results.
    
    Args:
        eval_path: Path to the evaluation results file. If None, uses a default path.
        figsize: Figure size as (width, height).
        
    Returns:
        Matplotlib figure.
    """
    if eval_path is None:
        eval_path = DATA_CONFIG["data_dir"] / "model_evaluation.csv"
    
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation file not found at {eval_path}")
    
    # Load evaluation results
    eval_df = pd.read_csv(eval_path)
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot actual vs predicted
    ax1.plot(eval_df["timestamp"], eval_df["actual_wait_time"], 'b-', label="Actual")
    ax1.plot(eval_df["timestamp"], eval_df["predicted_wait_time"], 'r--', label="Predicted")
    ax1.set_title("Actual vs Predicted Wait Times")
    ax1.set_ylabel("Wait Time (minutes)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot error
    ax2.bar(eval_df["timestamp"], eval_df["absolute_error"], color='orange', alpha=0.7)
    ax2.set_title("Prediction Error")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Absolute Error (minutes)")
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    evaluate_with_live_data()
