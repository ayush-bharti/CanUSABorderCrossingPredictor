"""
Test the Prophet model with synthetic data.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.data.data_loader import preprocess_data, add_features, split_data
from border_predictor.utils.helpers import calculate_error_metrics


def generate_synthetic_data(
    start_date: str = "2023-01-01",
    periods: int = 720,  # 30 days x 24 hours
    freq: str = "H"
) -> pd.DataFrame:
    """
    Generate synthetic border crossing wait time data.
    
    Args:
        start_date: Start date for the synthetic data
        periods: Number of periods to generate
        freq: Frequency of the data
    
    Returns:
        DataFrame with synthetic data
    """
    print(f"Generating synthetic data from {start_date} with {periods} {freq} periods...")
    
    # Create date range
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Create base wait time (in minutes)
    base_wait_time = 15
    
    # Add hourly pattern (rush hours have higher wait times)
    hour_effect = np.array([
        5, 5, 5, 5, 5, 10,  # 0-5 AM
        20, 30, 35, 25, 20, 15,  # 6-11 AM
        25, 30, 25, 20, 30, 40,  # 12-5 PM
        30, 25, 15, 10, 5, 5  # 6-11 PM
    ])
    
    # Add day of week pattern (weekends have higher wait times)
    day_effect = np.array([10, 5, 5, 10, 15, 25, 20])  # Mon-Sun
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': date_range,
        'wait_time': base_wait_time
    })
    
    # Add hour effect
    df['wait_time'] += hour_effect[df['datetime'].dt.hour]
    
    # Add day of week effect
    df['wait_time'] += day_effect[df['datetime'].dt.dayofweek]
    
    # Add random noise
    np.random.seed(42)
    df['wait_time'] += np.random.normal(0, 5, size=len(df))
    
    # Ensure wait times are positive
    df['wait_time'] = np.maximum(df['wait_time'], 1)
    
    # Round to integers
    df['wait_time'] = df['wait_time'].round().astype(int)
    
    return df


def test_prophet_model():
    """
    Test the Prophet model with synthetic data.
    """
    print("\nTesting Prophet model with synthetic data...")
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_data()
    
    # Preprocess data for Prophet
    prophet_df = preprocess_data(synthetic_df)
    
    # Add features
    prophet_df = add_features(prophet_df)
    
    # Split data
    train_df, test_df = split_data(prophet_df, test_size=0.2)
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Testing data: {len(test_df)} samples")
    
    # Create and train model
    print("\nTraining Prophet model...")
    model = BorderProphetModel()
    model.fit(train_df[['ds', 'y']])
    
    # Make predictions on test set
    print("Generating predictions...")
    future_df = test_df[['ds']].copy()
    forecast = model.predict(future_df=future_df)
    
    # Evaluate model
    print("\nEvaluating model...")
    actual = test_df['y']
    predicted = forecast.loc[forecast['ds'].isin(test_df['ds']), 'yhat']
    
    metrics = calculate_error_metrics(actual, predicted)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create a simple plot to visualize results
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual values
    ax.plot(test_df['ds'], test_df['y'], 'b-', label='Actual')
    
    # Plot predicted values
    ax.plot(forecast['ds'], forecast['yhat'], 'r-', label='Predicted')
    
    # Plot confidence intervals
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='r',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Wait Time (minutes)')
    ax.set_title('Prophet Model: Actual vs. Predicted Wait Times')
    ax.legend()
    ax.grid(True)
    
    # Save the plot
    fig.tight_layout()
    plot_path = Path(__file__).parent.parent / "data" / "test_plot.png"
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    return metrics


if __name__ == "__main__":
    test_prophet_model()
