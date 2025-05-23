"""Script to test model predictions for future dates and times."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import numpy as np
import os
from pathlib import Path

from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.config.config import MODEL_CONFIG, BORDER_CONFIG, DATA_CONFIG, LIVE_DATA_CONFIG


def load_model(model_path=None):
    """Load the trained model."""
    if model_path is None:
        model_path = MODEL_CONFIG["model_path"]
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def adjust_prediction_with_recent_data(prediction: float, recent_data: pd.DataFrame, adjustment_weight: float = 0.7) -> float:
    """
    Adjust a prediction based on recent actual wait times.
    
    Args:
        prediction: The model's raw prediction
        recent_data: DataFrame with recent wait time data, must have 'ds' and 'y' columns
        adjustment_weight: Weight to give to recent data (0-1), higher values favor recent data
    
    Returns:
        Adjusted prediction
    """
    if recent_data is None or len(recent_data) == 0:
        return prediction
    
    # Sort by timestamp and get the most recent values
    recent_data = recent_data.sort_values('ds')
    
    # Get the most recent 5 measurements (or fewer if not available)
    num_recent = min(5, len(recent_data))
    most_recent = recent_data.tail(num_recent)
    
    # Calculate the average of recent wait times
    recent_avg = most_recent['y'].mean()
    
    # Apply time-based weighting (more recent = higher weight)
    weights = np.linspace(0.5, 1.0, num_recent)
    weighted_avg = np.average(most_recent['y'].values, weights=weights)
    
    # Calculate time since last observation
    last_time = pd.to_datetime(most_recent['ds'].max())
    now = datetime.now()
    minutes_since_last = (now - last_time).total_seconds() / 60
    
    # Adjust weight based on recency (reduce weight if data is old)
    if minutes_since_last > 30:
        # Reduce adjustment_weight if data is more than 30 minutes old
        adjustment_weight = adjustment_weight * (1 - min(0.5, (minutes_since_last - 30) / 60))
    
    # Calculate adjusted prediction as weighted average of model prediction and recent data
    adjusted_prediction = (1 - adjustment_weight) * prediction + adjustment_weight * weighted_avg
    
    # Ensure prediction is non-negative
    adjusted_prediction = max(0, adjusted_prediction)
    
    return adjusted_prediction


def load_recent_live_data():
    """
    Load recent live data for prediction adjustment.
    
    Returns:
        DataFrame with recent live data or None if not available
    """
    live_data_path = DATA_CONFIG["live_data_path"]
    recent_data = None
    
    if os.path.exists(live_data_path):
        try:
            recent_data = pd.read_csv(live_data_path)
            if len(recent_data) > 0:
                # Convert timestamps to datetime
                recent_data['ds'] = pd.to_datetime(recent_data['ds'])
                print(f"Loaded {len(recent_data)} recent data points for prediction adjustment")
            else:
                print("No recent data points available for prediction adjustment")
        except Exception as e:
            print(f"Warning: Could not load recent data for adjustment: {e}")
    else:
        print(f"Live data file not found at {live_data_path}")
    
    return recent_data


def predict_for_specific_time(model, target_datetime, display_range_hours=12, apply_live_adjustment=True):
    """Generate predictions for a specific date and time."""
    # Create a DataFrame with future dates for prediction
    start_time = target_datetime - timedelta(hours=display_range_hours//2)
    end_time = target_datetime + timedelta(hours=display_range_hours//2)
    
    # Create hourly timestamps between start and end times
    future_times = pd.date_range(start=start_time, end=end_time, freq='H')
    future_df = pd.DataFrame({"ds": future_times})
    
    # Generate predictions
    forecast = model.predict(future_df=future_df)
    
    # Print the prediction for the target time
    target_prediction = forecast[forecast['ds'] == pd.Timestamp(target_datetime)]
    
    if len(target_prediction) > 0:
        yhat = target_prediction['yhat'].values[0]
        yhat_lower = target_prediction['yhat_lower'].values[0]
        yhat_upper = target_prediction['yhat_upper'].values[0]
        
        # Apply adjustment from live data if requested
        adjusted_yhat = yhat
        adjustment_effect = 0
        
        if apply_live_adjustment:
            # Load recent live data
            recent_data = load_recent_live_data()
            
            # Apply adjustment if data is available
            if recent_data is not None and len(recent_data) > 0:
                adjusted_yhat = adjust_prediction_with_recent_data(yhat, recent_data)
                adjustment_effect = adjusted_yhat - yhat
        
        print(f"\nPrediction for {target_datetime}:")
        print(f"  Raw model prediction: {yhat:.2f} minutes")
        
        if apply_live_adjustment:
            print(f"  Live data adjustment: {adjustment_effect:.2f} minutes")
            print(f"  Adjusted wait time: {adjusted_yhat:.2f} minutes")
        else:
            print(f"  Wait time: {yhat:.2f} minutes")
            
        print(f"  95% Confidence interval: [{yhat_lower:.2f}, {yhat_upper:.2f}] minutes")
        
        # If prediction is negative, warn about potential model issues
        if yhat < 0:
            print("\nWARNING: Negative prediction detected!")
            print("This suggests the model needs retraining with constraints or further tuning.")
            
        # Update the target prediction with the adjusted value if adjustment was applied
        if apply_live_adjustment:
            yhat = adjusted_yhat
    else:
        print(f"No prediction found for {target_datetime}")
    
    # Plot the predictions around the target time
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Wait Time')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     alpha=0.3, label='95% Confidence Interval')
    
    # Add a horizontal line at y=0 to clearly see negative predictions
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Wait Time')
    
    # Highlight the target time
    if len(target_prediction) > 0:
        plt.scatter([target_datetime], [yhat], color='green', s=100, 
                   label=f'Target Time: {yhat:.2f} mins')
    
    plt.title(f'Border Wait Time Predictions Around {target_datetime}')
    plt.xlabel('Date and Time')
    plt.ylabel('Wait Time (minutes)')
    plt.gcf().autofmt_xdate()  # Auto-format dates for better readability
    date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for better readability
    plt.tight_layout()
    
    return forecast, plt.gcf()


if __name__ == "__main__":
    # Load the model
    model = load_model()
    
    # Target datetime for tomorrow at noon
    tomorrow_noon = datetime.now() + timedelta(days=1)
    tomorrow_noon = tomorrow_noon.replace(hour=12, minute=0, second=0, microsecond=0)
    
    # Get predictions
    forecast, fig = predict_for_specific_time(model, tomorrow_noon)
    
    # Load recent live data for adjustments
    recent_data = load_recent_live_data()
    
    # Also check predictions for various times of the day
    print("\nPredictions throughout tomorrow:")
    times = [6, 8, 12, 17, 20]  # Hours to check (6am, 8am, noon, 5pm, 8pm)
    
    for hour in times:
        target_time = tomorrow_noon.replace(hour=hour)
        target_pred = forecast[forecast['ds'] == pd.Timestamp(target_time)]
        if len(target_pred) > 0:
            raw_pred_value = max(0, target_pred['yhat'].values[0])  # Prevent negative values
            
            # Apply adjustment if data is available
            adjusted_pred_value = raw_pred_value
            if recent_data is not None and len(recent_data) > 0:
                adjusted_pred_value = adjust_prediction_with_recent_data(raw_pred_value, recent_data)
                adjustment = adjusted_pred_value - raw_pred_value
                print(f"  {hour:02d}:00 - Raw: {raw_pred_value:.2f} min, Adjusted: {adjusted_pred_value:.2f} min (Δ: {adjustment:+.2f})")
            else:
                print(f"  {hour:02d}:00 - {raw_pred_value:.2f} minutes")
    
    # Show the plot
    plt.show()
