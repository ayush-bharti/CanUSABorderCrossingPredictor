"""Runner script for Phase 2 of the border crossing prediction project."""
from typing import Dict, List, Optional, Union
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import pickle
from datetime import datetime, timedelta

from border_predictor.data.data_loader import load_processed_data
from border_predictor.data.live_data_loader import (
    get_current_wait_time,
    prepare_live_data_for_model,
    save_live_data
)
from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.models.train_model import train_model
from border_predictor.models.evaluate_live import evaluate_with_live_data, plot_evaluation_history
from border_predictor.config.config import DATA_CONFIG, MODEL_CONFIG, BORDER_CONFIG, LIVE_DATA_CONFIG


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_monthly_data_cli(args):
    """
    Train model with monthly data from CLI arguments.
    
    Args:
        args: CLI arguments
    """
    monthly_path = Path(args.monthly_data) if args.monthly_data else None
    model_path = Path(args.model) if args.model else None
    
    logger.info(f'Training model with monthly data from {monthly_path if monthly_path else DATA_CONFIG["monthly_data_path"]}')
    model, metrics = train_model(data_path=monthly_path, model_path=model_path)
    
    # Print metrics
    logger.info("Model evaluation metrics:")
    for metric, value in metrics.mean().items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f'Model saved to {model_path if model_path else MODEL_CONFIG["model_path"]}')


def evaluate_live_data_cli(args):
    """
    Evaluate model with live data from CLI arguments.
    
    Args:
        args: CLI arguments
    """
    model_path = Path(args.model) if args.model else None
    crossing_name = args.crossing_name if args.crossing_name else BORDER_CONFIG['crossing_name']
    lane_type = args.lane_type if args.lane_type else BORDER_CONFIG.get('lane_type', 'standard')
    forecast_hours = args.forecast_hours if args.forecast_hours else 24
    
    logger.info(f'Evaluating model with live data for {crossing_name} ({lane_type} lanes)')
    
    try:
        # Load model
        if model_path and model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f'Loaded model from {model_path}')
        else:
            model = None  # Will be loaded from default location
            logger.info('Using default model path')
            
        # Evaluate with live data
        eval_results, error = evaluate_with_live_data(
            model=model,
            forecast_hours=forecast_hours,
            crossing_name=crossing_name,
            lane_type=lane_type
        )
        
        logger.info(f'Evaluation complete with error: {error:.2f} minutes')
        
        # Try to plot evaluation history if available
        try:
            fig = plot_evaluation_history()
            plt.show()
        except FileNotFoundError:
            logger.warning('Not enough evaluation data to plot history')
        
    except Exception as e:
        logger.error(f'Error evaluating with live data: {e}')


def main():
    """Main function for Phase 2 runner."""
    parser = argparse.ArgumentParser(description='Border Crossing Prediction Phase 2')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train with monthly data command
    train_parser = subparsers.add_parser('train', help='Train model with monthly data')
    train_parser.add_argument('--monthly-data', type=str, help='Path to monthly data CSV file')
    train_parser.add_argument('--model', type=str, help='Path to save trained model')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data for testing')
    
    # Evaluate with live data command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model with live data')
    eval_parser.add_argument('--model', type=str, help='Path to trained model file')
    eval_parser.add_argument('--crossing-name', type=str, help='Border crossing name')
    eval_parser.add_argument('--lane-type', choices=['standard', 'nexus'], default='standard', 
                      help='Lane type (standard or nexus)')
    eval_parser.add_argument('--forecast-hours', type=int, default=24, 
                      help='Number of hours to forecast')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_with_monthly_data_cli(args)
    elif args.command == 'evaluate':
        evaluate_live_data_cli(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
