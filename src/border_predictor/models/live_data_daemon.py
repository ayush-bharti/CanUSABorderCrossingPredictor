"""Daemon process for continuous live data collection and model fine-tuning."""
import time
import logging
from datetime import datetime
from pathlib import Path
import signal
import sys
from typing import Optional

from border_predictor.models.prophet_model import ProphetModel
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
        model: ProphetModel,
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
        
        logger.info("Starting live data collection daemon...")
        
        while self.running:
            try:
                # Collect live data
                wait_time, info = self.tuner.collect_live_data()
                logger.info(f"Collected wait time: {wait_time} minutes")
                
                # Check if we should retrain
                if self.tuner.should_retrain():
                    logger.info("Retraining model with new data...")
                    metrics, new_model = self.tuner.fine_tune_model()
                    if metrics:  # Only update if fine-tuning was successful
                        self.model = new_model
                        logger.info(f"Model fine-tuned successfully. New metrics: {metrics}")
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in daemon process: {e}")
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Daemon process stopped.")

def main():
    """Main entry point for the daemon process."""
    try:
        # Load the latest model
        model = ProphetModel.load_latest()
        
        # Create and run the daemon
        daemon = LiveDataDaemon(model)
        daemon.run()
        
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
