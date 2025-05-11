"""Helper functions for border crossing prediction."""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import holidays

from border_predictor.config.config import BORDER_CONFIG


def get_holidays(years: List[int]) -> pd.DataFrame:
    """
    Get holidays for the specified countries and years.
    
    Args:
        years: List of years to get holidays for.
    
    Returns:
        DataFrame with holidays information.
    """
    # Initialize empty list to store holidays
    all_holidays = []
    
    # Get holidays for each country
    for country in BORDER_CONFIG["countries"]:
        if country == "US":
            country_holidays = holidays.US(years=years)
        elif country == "Canada":
            country_holidays = holidays.Canada(years=years)
        else:
            continue
        
        # Convert holidays to DataFrame
        for date, name in country_holidays.items():
            all_holidays.append({
                'ds': date,
                'holiday': f"{country}_{name.replace(' ', '_')}",
                'country': country
            })
    
    # Convert to DataFrame
    holidays_df = pd.DataFrame(all_holidays)
    
    return holidays_df


def calculate_error_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate error metrics for model evaluation.
    
    Args:
        actual: Series of actual values.
        predicted: Series of predicted values.
    
    Returns:
        Dictionary of error metrics.
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(predicted - actual))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    
    # Mean Squared Error
    mse = np.mean((predicted - actual) ** 2)
    
    # R-squared
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2
    }


def format_time(minutes: float) -> str:
    """
    Format minutes as hours and minutes.
    
    Args:
        minutes: Time in minutes.
    
    Returns:
        Formatted time string.
    """
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours > 0:
        return f"{hours} hr {mins} min"
    else:
        return f"{mins} min"


def get_time_features(dt: Union[datetime, pd.Timestamp]) -> Dict[str, Union[int, bool]]:
    """
    Extract time features from a datetime object.
    
    Args:
        dt: Datetime object.
    
    Returns:
        Dictionary of time features.
    """
    return {
        'hour': dt.hour,
        'day_of_week': dt.dayofweek,
        'month': dt.month,
        'day': dt.day,
        'year': dt.year,
        'is_weekend': dt.dayofweek >= 5,
        'time_of_day': get_time_of_day(dt.hour)
    }


def get_time_of_day(hour: int) -> str:
    """
    Get time of day category based on hour.
    
    Args:
        hour: Hour of day (0-23).
    
    Returns:
        Time of day category.
    """
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


def plot_wait_time_by_hour(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot average wait time by hour of day.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure.
    """
    # Extract hour from datetime
    df['hour'] = df['ds'].dt.hour
    
    # Calculate average wait time by hour
    hourly_avg = df.groupby('hour')['y'].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    ax.bar(hourly_avg['hour'], hourly_avg['y'], color='skyblue')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Wait Time (minutes)')
    ax.set_title('Average Border Crossing Wait Time by Hour of Day')
    ax.set_xticks(range(24))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig


def plot_wait_time_by_day(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot average wait time by day of week.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure.
    """
    # Extract day of week from datetime
    df['day_of_week'] = df['ds'].dt.dayofweek
    
    # Calculate average wait time by day of week
    daily_avg = df.groupby('day_of_week')['y'].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax.bar(daily_avg['day_of_week'], daily_avg['y'], color='lightgreen')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Wait Time (minutes)')
    ax.set_title('Average Border Crossing Wait Time by Day of Week')
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig


def plot_wait_time_by_month(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot average wait time by month.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns.
        figsize: Figure size as (width, height).
    
    Returns:
        Matplotlib figure.
    """
    # Extract month from datetime
    df['month'] = df['ds'].dt.month
    
    # Calculate average wait time by month
    monthly_avg = df.groupby('month')['y'].mean().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.bar(monthly_avg['month'], monthly_avg['y'], color='salmon')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Wait Time (minutes)')
    ax.set_title('Average Border Crossing Wait Time by Month')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig
