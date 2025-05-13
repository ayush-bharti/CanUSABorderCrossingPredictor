"""Utility functions for calculating model performance metrics."""
from typing import Dict
import numpy as np
import pandas as pd


def calculate_metrics(actual: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance metrics for model predictions.
    
    Args:
        actual: DataFrame with actual values, must have 'ds' and 'y' columns
        forecast: DataFrame with forecasted values, must have 'ds' and 'yhat' columns
        
    Returns:
        Dictionary with metrics (MAE, RMSE, MAPE)
    """
    # Merge actual and predicted values
    df = pd.merge(actual, forecast[['ds', 'yhat']], on='ds', how='inner')
    
    if len(df) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan
        }
    
    # Calculate metrics
    mae = np.mean(np.abs(df['y'] - df['yhat']))
    rmse = np.sqrt(np.mean((df['y'] - df['yhat'])**2))
    
    # Calculate MAPE, handling zero values
    non_zero_mask = df['y'] != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((df.loc[non_zero_mask, 'y'] - 
                              df.loc[non_zero_mask, 'yhat']) / 
                              df.loc[non_zero_mask, 'y'])) * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
