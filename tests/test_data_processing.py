"""
Test the data processing pipeline without relying on Prophet model fitting.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from border_predictor.data.data_loader import preprocess_data, add_features, split_data
from border_predictor.utils.helpers import calculate_error_metrics, get_holidays


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


def test_data_processing():
    """
    Test the data processing pipeline.
    """
    print("\nTesting data processing pipeline...")
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_data()
    print(f"Generated data shape: {synthetic_df.shape}")
    print(f"Generated data head:\n{synthetic_df.head()}")
    
    # Preprocess data for Prophet
    prophet_df = preprocess_data(synthetic_df)
    print(f"\nPreprocessed data shape: {prophet_df.shape}")
    print(f"Preprocessed data head:\n{prophet_df.head()}")
    
    # Verify the preprocessed data has the expected columns
    assert 'ds' in prophet_df.columns, "Column 'ds' missing from preprocessed data"
    assert 'y' in prophet_df.columns, "Column 'y' missing from preprocessed data"
    
    # Add features
    featured_df = add_features(prophet_df)
    print(f"\nFeatured data shape: {featured_df.shape}")
    print(f"Featured data columns: {', '.join(featured_df.columns)}")
    
    # Verify the featured data has the expected columns
    expected_features = ['hour', 'day_of_week', 'month', 'is_weekend']
    for feature in expected_features:
        assert feature in featured_df.columns, f"Feature '{feature}' missing from featured data"
    
    # Split data
    train_df, test_df = split_data(featured_df, test_size=0.2)
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    # Verify the split was done correctly
    assert len(train_df) + len(test_df) == len(featured_df), "Data split has incorrect total size"
    assert abs(len(test_df) / len(featured_df) - 0.2) < 0.01, "Test set size does not match specified test_size"
    
    # Test holiday generation
    print("\nTesting holiday generation...")
    years = [2023, 2024]
    holidays_df = get_holidays(years)
    print(f"Generated holidays for years {years}")
    print(f"Holiday data shape: {holidays_df.shape}")
    print(f"Holiday data sample:\n{holidays_df.head(3)}")
    
    # Verify the holidays data has the expected columns
    assert 'ds' in holidays_df.columns, "Column 'ds' missing from holidays data"
    assert 'holiday' in holidays_df.columns, "Column 'holiday' missing from holidays data"
    assert 'country' in holidays_df.columns, "Column 'country' missing from holidays data"
    
    # Save sample data for future use
    data_dir = Path(__file__).parent.parent / "data"
    synthetic_df.to_csv(data_dir / "sample_data.csv", index=False)
    print(f"\nSaved sample data to {data_dir / 'sample_data.csv'}")
    
    print("\nAll data processing tests passed successfully!")
    return True


if __name__ == "__main__":
    test_data_processing()
