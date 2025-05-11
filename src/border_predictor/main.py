"""Main script for border crossing prediction."""
from typing import Dict, List, Optional, Union
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from border_predictor.data.data_loader import (
    load_historical_data,
    load_current_data,
    preprocess_data,
    add_features,
    save_processed_data,
    split_data
)
from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.utils.helpers import (
    calculate_error_metrics,
    format_time,
    plot_wait_time_by_hour,
    plot_wait_time_by_day,
    plot_wait_time_by_month
)
from border_predictor.config.config import DATA_CONFIG, MODEL_CONFIG, BORDER_CONFIG


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    historical_data_path: Optional[Path] = None,
    current_data_path: Optional[Path] = None,
    save_model: bool = True,
    plot_results: bool = True
) -> BorderProphetModel:
    """
    Train the Prophet model on historical data.
    
    Args:
        historical_data_path: Path to historical data file.
        current_data_path: Path to current data file.
        save_model: Whether to save the trained model.
        plot_results: Whether to generate and display plots.
    
    Returns:
        Trained BorderProphetModel instance.
    """
    logger.info("Loading and preprocessing data...")
    
    try:
        # Load historical data
        historical_df = load_historical_data(historical_data_path)
        
        # Load current data if available
        current_df = None
        if current_data_path:
            try:
                current_df = load_current_data(current_data_path)
            except FileNotFoundError:
                logger.warning(f"Current data file not found at {current_data_path}")
        
        # Preprocess data
        prophet_df = preprocess_data(historical_df, current_df)
        
        # Add features
        prophet_df = add_features(prophet_df)
        
        # Save processed data
        save_processed_data(prophet_df)
        
        # Split data
        train_df, test_df = split_data(prophet_df, test_size=0.2)
        
        logger.info(f"Data processed: {len(train_df)} training samples, {len(test_df)} testing samples")
        
        # Create and train model
        logger.info("Training Prophet model...")
        model = BorderProphetModel()
        model.fit(train_df[['ds', 'y']])
        
        # Make predictions on test set
        logger.info("Generating predictions...")
        future_df = test_df[['ds']].copy()
        forecast = model.predict(future_df=future_df)
        
        # Evaluate model
        logger.info("Evaluating model...")
        actual = test_df['y']
        predicted = forecast.loc[forecast['ds'].isin(test_df['ds']), 'yhat']
        
        metrics = calculate_error_metrics(actual, predicted)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save model if requested
        if save_model:
            logger.info("Saving model...")
            model.save_model()
        
        # Generate plots if requested
        if plot_results:
            logger.info("Generating plots...")
            
            # Plot forecast
            model.plot_forecast(forecast)
            plt.savefig(Path(DATA_CONFIG["processed_data_path"]).parent / "forecast_plot.png")
            
            # Plot components
            model.plot_components(forecast)
            plt.savefig(Path(DATA_CONFIG["processed_data_path"]).parent / "components_plot.png")
            
            # Plot additional visualizations
            plot_wait_time_by_hour(prophet_df)
            plt.savefig(Path(DATA_CONFIG["processed_data_path"]).parent / "hourly_plot.png")
            
            plot_wait_time_by_day(prophet_df)
            plt.savefig(Path(DATA_CONFIG["processed_data_path"]).parent / "daily_plot.png")
            
            plot_wait_time_by_month(prophet_df)
            plt.savefig(Path(DATA_CONFIG["processed_data_path"]).parent / "monthly_plot.png")
        
        logger.info("Model training and evaluation complete!")
        return model
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
    

def predict_wait_time(
    target_date: str,
    model_path: Optional[Path] = None,
    historical_data_path: Optional[Path] = None,
    current_wait_time: Optional[float] = None
) -> Dict:
    """
    Predict border crossing wait time for a specific date and time.
    
    Args:
        target_date: Target date and time for prediction (format: 'YYYY-MM-DD HH:MM:SS').
        model_path: Path to the trained model file.
        historical_data_path: Path to historical data file (used if model not found).
        current_wait_time: Current wait time in minutes (if available).
    
    Returns:
        Dictionary with prediction results.
    """
    logger.info(f"Predicting wait time for {target_date}...")
    
    try:
        # Parse target date
        target_date = pd.Timestamp(target_date)
        
        # Load or train model
        model = BorderProphetModel()
        
        try:
            # Try to load existing model
            model.load_model(model_path)
            logger.info("Model loaded successfully")
        except (FileNotFoundError, ValueError):
            # Train new model if not found
            logger.warning("Model not found, training new model...")
            if historical_data_path is None:
                historical_data_path = DATA_CONFIG["historical_data_path"]
            model = train_model(historical_data_path=historical_data_path, save_model=True, plot_results=False)
        
        # Create future dataframe for prediction
        future_df = pd.DataFrame({'ds': [target_date]})
        
        # Add current wait time as regressor if available
        if current_wait_time is not None:
            # This would require modifying the model to include this regressor
            # For now, we'll just log it
            logger.info(f"Current wait time: {current_wait_time} minutes")
        
        # Make prediction
        forecast = model.predict(future_df=future_df)
        
        # Extract prediction
        prediction = forecast.iloc[0]
        
        # Format results
        result = {
            'target_date': target_date.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_wait_time': prediction['yhat'],
            'predicted_wait_time_formatted': format_time(prediction['yhat']),
            'lower_bound': prediction['yhat_lower'],
            'upper_bound': prediction['yhat_upper'],
            'crossing_name': BORDER_CONFIG['crossing_name']
        }
        
        logger.info(f"Predicted wait time: {result['predicted_wait_time_formatted']} " +
                   f"(range: {format_time(result['lower_bound'])} - {format_time(result['upper_bound'])})")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def main():
    """Main function to run the border crossing prediction."""
    parser = argparse.ArgumentParser(description='Border Crossing Wait Time Prediction')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--historical-data', type=str, help='Path to historical data file')
    train_parser.add_argument('--current-data', type=str, help='Path to current data file')
    train_parser.add_argument('--no-save', action='store_true', help='Do not save the model')
    train_parser.add_argument('--no-plots', action='store_true', help='Do not generate plots')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict wait time')
    predict_parser.add_argument('--date', type=str, required=True, 
                               help='Target date and time (format: YYYY-MM-DD HH:MM:SS)')
    predict_parser.add_argument('--model', type=str, help='Path to model file')
    predict_parser.add_argument('--historical-data', type=str, help='Path to historical data file')
    predict_parser.add_argument('--current-wait', type=float, help='Current wait time in minutes')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Convert string paths to Path objects
        historical_data_path = Path(args.historical_data) if args.historical_data else None
        current_data_path = Path(args.current_data) if args.current_data else None
        
        # Train model
        train_model(
            historical_data_path=historical_data_path,
            current_data_path=current_data_path,
            save_model=not args.no_save,
            plot_results=not args.no_plots
        )
    
    elif args.command == 'predict':
        # Convert string paths to Path objects
        model_path = Path(args.model) if args.model else None
        historical_data_path = Path(args.historical_data) if args.historical_data else None
        
        # Predict wait time
        predict_wait_time(
            target_date=args.date,
            model_path=model_path,
            historical_data_path=historical_data_path,
            current_wait_time=args.current_wait
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
