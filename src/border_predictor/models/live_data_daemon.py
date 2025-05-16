"""Daemon process for continuous live data collection and model fine-tuning."""
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import signal
import sys
from typing import Optional

from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.models.live_model_tuner import LiveModelTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveDataDaemon:
    """Daemon process for continuous live data collection and model fine-tuning."""
    
    def __init__(
        self,
        model: BorderProphetModel,
        collection_interval: int = 15 * 60,  # 15 minutes in seconds
        live_data_path: Optional[Path] = None
    ):
        """
        Initialize the daemon process.
        
        Args:
            model: Trained Prophet model to fine-tune
            collection_interval: Seconds between data collection (default: 15 minutes)
            live_data_path: Path to store live data
        """
        self.model = model
        self.collection_interval = collection_interval
        self.tuner = LiveModelTuner(model, live_data_path)
        self.running = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, signum: int, frame) -> None:
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.running = False
    
    def run(self) -> None:
        """Run the daemon process."""
        self.running = True
        last_collection = datetime.now()
        collection_count = 0
        
        logger.info("Starting live data collection daemon...")
        logger.info(f"Collection interval: {self.collection_interval//60} minutes")
        logger.info(f"Live data path: {self.tuner.live_data_path}")
        logger.info(f"Retraining threshold: {self.tuner.retraining_threshold} data points")
        
        while self.running:
            try:
                collection_start = datetime.now()
                collection_count += 1
                
                # Log collection attempt
                logger.info("-" * 60)
                logger.info(f"Data collection cycle #{collection_count} started at {collection_start.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Collect live data
                wait_time, info = self.tuner.collect_live_data()
                logger.info(f"Collected wait time: {wait_time} minutes")
                logger.info(f"Border crossing status: {info.get('port_status', 'unknown')}")
                logger.info(f"Last border update time: {info.get('update_time', 'unknown')}")
                
                # Check if we should retrain
                should_retrain = self.tuner.should_retrain()
                current_data_points = len(self.tuner.live_data)
                logger.info(f"Current data points: {current_data_points} / {self.tuner.retraining_threshold} required for retraining")
                
                if should_retrain:
                    logger.info("Retraining model with new data...")
                    retrain_start = datetime.now()
                    metrics, new_model = self.tuner.fine_tune_model()
                    
                    if metrics:  # Only update if fine-tuning was successful
                        self.model = new_model
                        retrain_duration = (datetime.now() - retrain_start).total_seconds()
                        logger.info(f"Model fine-tuned successfully in {retrain_duration:.2f} seconds")
                        logger.info(f"New metrics: MAE={metrics.get('mae', 'N/A')}, RMSE={metrics.get('rmse', 'N/A')}, MAPE={metrics.get('mape', 'N/A')}")
                    else:
                        logger.warning("Fine-tuning did not produce valid metrics - keeping existing model")
                
                # Calculate time until next collection
                collection_duration = (datetime.now() - collection_start).total_seconds()
                sleep_time = max(1, self.collection_interval - collection_duration)
                next_collection = datetime.now() + timedelta(seconds=sleep_time)
                
                logger.info(f"Collection cycle completed in {collection_duration:.2f} seconds")
                logger.info(f"Next collection scheduled at {next_collection.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("-" * 60)
                
                # Sleep until next collection
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in daemon process: {e}")
                logger.error("Detailed traceback:")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("Will retry in 60 seconds...")
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Daemon process stopped.")

def main():
    """Main entry point for the daemon process."""
    try:
        # Load the latest model
        model = BorderProphetModel.load_latest()
        
        # Create and run the daemon
        daemon = LiveDataDaemon(model)
        daemon.run()
        
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
