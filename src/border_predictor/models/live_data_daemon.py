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
        collection_count = 0
        
        logger.info("Starting live data collection daemon...")
        logger.info(f"Collection interval: {self.collection_interval//60} minutes")
        logger.info(f"Live data path: {self.tuner.live_data_path}")
        logger.info(f"Retraining threshold: {self.tuner.retraining_threshold} data points")
        
        # Initial collection
        logger.info("Starting initial data collection...")
        try:
            wait_time, info = self.tuner.collect_live_data()
            logger.info(f"Initial collection successful: {wait_time} minutes")
            last_collection = datetime.now()
        except Exception as e:
            logger.error(f"Initial collection failed: {e}")
            last_collection = datetime.now() - timedelta(seconds=self.collection_interval)  # Force immediate retry
        
        while self.running:
            try:
                current_time = datetime.now()
                time_since_last = (current_time - last_collection).total_seconds()
                
                if time_since_last < self.collection_interval:
                    sleep_time = max(1, self.collection_interval - time_since_last)
                    logger.debug(f"Waiting {sleep_time:.1f} seconds until next collection")
                    time.sleep(sleep_time)
                    continue
                
                collection_start = current_time
                collection_count += 1
                last_collection = current_time  # Update last collection time
                
                collection_count += 1
                logger.info("-" * 60)
                logger.info(f"Data collection cycle #{collection_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Collect live data
                wait_time, info = self.tuner.collect_live_data()
                last_collection = current_time  # Update last collection time
                
                logger.info(f"Collected wait time: {wait_time} minutes")
                logger.info(f"Border crossing status: {info.get('port_status', 'unknown')}")
                logger.info(f"Last border update time: {info.get('update_time', 'unknown')}")
                logger.info(f"Next collection scheduled for: {(current_time + timedelta(seconds=self.collection_interval)).strftime('%Y-%m-%d %H:%M:%S')}")
                
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
                
                # Log completion
                collection_duration = (datetime.now() - collection_start).total_seconds()
                next_collection = last_collection + timedelta(seconds=self.collection_interval)
                
                logger.info(f"Collection cycle completed in {collection_duration:.2f} seconds")
                logger.info(f"Next collection scheduled at {next_collection.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("-" * 60)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
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
        # Create and load the latest model
        model = BorderProphetModel()
        model.load_model()  # Uses default path from config
        
        # Create and run the daemon
        daemon = LiveDataDaemon(model)
        daemon.run()
        
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
