"""Script to test model predictions for future dates and times."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
from pathlib import Path

from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.config.config import MODEL_CONFIG, BORDER_CONFIG


def load_model(model_path=None):
    """Load the trained model."""
    if model_path is None:
        model_path = MODEL_CONFIG["model_path"]
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def predict_for_specific_time(model, target_datetime, display_range_hours=12):
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
        
        print(f"\nPrediction for {target_datetime}:")
        print(f"  Wait time: {yhat:.2f} minutes")
        print(f"  95% Confidence interval: [{yhat_lower:.2f}, {yhat_upper:.2f}] minutes")
        
        # If prediction is negative, warn about potential model issues
        if yhat < 0:
            print("\nWARNING: Negative prediction detected!")
            print("This suggests the model needs retraining with constraints or further tuning.")
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
    
    # Also check predictions for various times of the day
    print("\nPredictions throughout tomorrow:")
    times = [6, 8, 12, 17, 20]  # Hours to check (6am, 8am, noon, 5pm, 8pm)
    
    for hour in times:
        target_time = tomorrow_noon.replace(hour=hour)
        target_pred = forecast[forecast['ds'] == pd.Timestamp(target_time)]
        if len(target_pred) > 0:
            pred_value = max(0, target_pred['yhat'].values[0])  # Prevent negative values
            print(f"  {hour:02d}:00 - {pred_value:.2f} minutes")
    
    # Show the plot
    plt.show()
