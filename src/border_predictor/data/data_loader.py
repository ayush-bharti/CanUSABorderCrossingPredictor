"""Data loading and processing functions for border crossing prediction."""
from typing import Dict, List, Optional, Tuple, Union
import os
import pandas as pd
import numpy as np
from pathlib import Path

from border_predictor.config.config import DATA_CONFIG


def load_historical_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load historical border crossing wait time data.
    
    Args:
        file_path: Path to the historical data file. If None, uses the default path
                  from DATA_CONFIG.
    
    Returns:
        DataFrame containing historical wait time data.
    
    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if file_path is None:
        file_path = DATA_CONFIG["historical_data_path"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Historical data file not found at {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return df


def load_current_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load current border crossing wait time data.
    
    Args:
        file_path: Path to the current data file. If None, uses the default path
                  from DATA_CONFIG.
    
    Returns:
        DataFrame containing current wait time data.
    
    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    if file_path is None:
        file_path = DATA_CONFIG["current_data_path"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Current data file not found at {file_path}")
    
    # Load data based on file extension
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return df


def preprocess_data(
    historical_df: pd.DataFrame, 
    current_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Preprocess data for Prophet model.
    
    Args:
        historical_df: DataFrame containing historical wait time data.
        current_df: Optional DataFrame containing current wait time data.
    
    Returns:
        Preprocessed DataFrame ready for Prophet model.
    """
    # Make a copy to avoid modifying the original
    df = historical_df.copy()
    
    # Ensure datetime column is properly formatted
    if 'datetime' in df.columns:
        datetime_col = 'datetime'
    elif 'date' in df.columns:
        datetime_col = 'date'
    elif 'time' in df.columns:
        datetime_col = 'time'
    else:
        # Try to find a datetime column
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols:
            datetime_col = datetime_cols[0]
        else:
            raise ValueError("No datetime column found in the data")
    
    # Convert to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Rename columns for Prophet (requires 'ds' for datetime and 'y' for target)
    wait_time_cols = [col for col in df.columns if 'wait' in col.lower() or 'time' in col.lower()]
    if not wait_time_cols:
        raise ValueError("No wait time column found in the data")
    
    wait_time_col = wait_time_cols[0]
    
    # Create Prophet-compatible DataFrame
    prophet_df = pd.DataFrame({
        'ds': df[datetime_col],
        'y': df[wait_time_col]
    })
    
    # Add current data if provided
    if current_df is not None:
        # Process current data similarly
        current_df = current_df.copy()
        
        # Find datetime and wait time columns
        current_datetime_cols = [col for col in current_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        current_wait_cols = [col for col in current_df.columns if 'wait' in col.lower() or 'time' in col.lower()]
        
        if current_datetime_cols and current_wait_cols:
            current_datetime_col = current_datetime_cols[0]
            current_wait_col = current_wait_cols[0]
            
            # Convert to datetime
            current_df[current_datetime_col] = pd.to_datetime(current_df[current_datetime_col])
            
            # Create Prophet-compatible DataFrame
            current_prophet_df = pd.DataFrame({
                'ds': current_df[current_datetime_col],
                'y': current_df[current_wait_col]
            })
            
            # Combine with historical data
            prophet_df = pd.concat([prophet_df, current_prophet_df]).drop_duplicates(subset=['ds'])
    
    # Sort by datetime
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Handle missing values
    prophet_df['y'] = prophet_df['y'].fillna(method='ffill')
    
    return prophet_df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional features to the dataset.
    
    Args:
        df: DataFrame with 'ds' and 'y' columns.
    
    Returns:
        DataFrame with additional features.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Extract time-based features
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['year'] = df['ds'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create time-of-day categories
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 12),
        (df['hour'] >= 12) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] < 24)
    ]
    categories = ['night', 'morning', 'afternoon', 'evening']
    df['time_of_day'] = pd.Categorical(
        np.select(conditions, categories),
        categories=categories,
        ordered=True
    )
    
    return df


def save_processed_data(df: pd.DataFrame, file_path: Optional[Path] = None) -> None:
    """
    Save processed data to a file.
    
    Args:
        df: Processed DataFrame to save.
        file_path: Path to save the processed data. If None, uses the default path
                  from DATA_CONFIG.
    """
    if file_path is None:
        file_path = DATA_CONFIG["processed_data_path"]
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    df.to_csv(file_path, index=False)
    
    print(f"Processed data saved to {file_path}")


def load_processed_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load processed data from a file.
    
    Args:
        file_path: Path to the processed data file. If None, uses the default path
                  from DATA_CONFIG.
    
    Returns:
        Processed DataFrame.
    
    Raises:
        FileNotFoundError: If the processed data file doesn't exist.
    """
    if file_path is None:
        file_path = DATA_CONFIG["processed_data_path"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    
    return pd.read_csv(file_path, parse_dates=['ds'])


def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    validation_size: Optional[float] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split data into training and testing sets, with an optional validation set.
    
    Args:
        df: DataFrame to split.
        test_size: Proportion of data to use for testing.
        validation_size: Optional proportion of data to use for validation.
    
    Returns:
        If validation_size is None, returns (train_df, test_df).
        Otherwise, returns (train_df, validation_df, test_df).
    """
    # Sort by datetime
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    test_idx = int(n * (1 - test_size))
    
    if validation_size is None:
        # Split into train and test
        train_df = df.iloc[:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()
        return train_df, test_df
    else:
        # Calculate validation index
        val_idx = int(n * (1 - test_size - validation_size))
        
        # Split into train, validation, and test
        train_df = df.iloc[:val_idx].copy()
        val_df = df.iloc[val_idx:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()
        return train_df, val_df, test_df
