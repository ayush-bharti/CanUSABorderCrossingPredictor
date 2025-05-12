"""Script to train the Prophet model using historical monthly data."""
import pandas as pd
from pathlib import Path
import pickle
import os

from border_predictor.config.config import MODEL_CONFIG, DATA_CONFIG
from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.data.data_loader import load_processed_data


def train_model(data_path: Path = None, model_path: Path = None, test_size: float = 0.2):
    """
    Train a Prophet model on historical border crossing data.
    
    Args:
        data_path: Path to the processed data file. If None, uses the monthly data path from config.
        model_path: Path to save the trained model. If None, uses the path from config.
        test_size: Proportion of data to use for testing.
        
    Returns:
        The trained model and evaluation metrics.
    """
    # Load and prepare data
    if data_path is None:
        data_path = DATA_CONFIG["monthly_data_path"]
    
    if model_path is None:
        model_path = MODEL_CONFIG["model_path"]
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert ds column to datetime if needed
    if 'ds' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ds']):
        df['ds'] = pd.to_datetime(df['ds'])
    
    # Split data for training and testing
    train_size = int(len(df) * (1 - test_size))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"Training data size: {len(train_df)}")
    print(f"Testing data size: {len(test_df)}")
    
    # Create and train model
    model = BorderProphetModel()
    model.fit(train_df)
    
    # Evaluate model on test data
    metrics = model.evaluate(df, initial='14 days', period='1 day', horizon='1 day')
    print("Model evaluation metrics:")
    for metric, value in metrics.mean().items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save model
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    return model, metrics


if __name__ == "__main__":
    train_model()
