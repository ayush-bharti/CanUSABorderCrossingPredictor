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
        if not self.running:
            # If we're already shutting down, just log and return
            logger.info(f"Received signal {signum} during shutdown. Already stopping...")
            return
            
        logger.info(f"Received signal {signum}. Initiating shutdown...")
        self.running = False
        
        # Set a flag to indicate we're in shutdown mode
        if not hasattr(self, '_shutdown_started'):
            self._shutdown_started = True
            self._shutdown_start_time = time.time()
            logger.info("Shutdown initiated at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Try to clean up resources
            try:
                if hasattr(self, 'tuner') and self.tuner:
                    logger.info("Cleaning up tuner resources...")
                    # Save any pending data
                    if hasattr(self.tuner, 'live_data') and not self.tuner.live_data.empty:
                        logger.info(f"Saving {len(self.tuner.live_data)} pending data points...")
                        self.tuner._save_live_data()
            except Exception as e:
                logger.error(f"Error during shutdown cleanup: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def run(self) -> None:
        """Run the daemon process with signal-safe sleep intervals."""
        self.running = True
        collection_count = 0
        SLEEP_INTERVAL = 15  # 15-second intervals for better signal handling
        
        try:
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
                        # Break down sleep into smaller intervals for better signal handling
                        sleep_remaining = max(1, self.collection_interval - time_since_last)
                        logger.debug(f"Waiting {sleep_remaining:.1f} seconds until next collection")
                        
                        while sleep_remaining > 0 and self.running:
                            sleep_chunk = min(SLEEP_INTERVAL, sleep_remaining)
                            time.sleep(sleep_chunk)
                            sleep_remaining -= sleep_chunk
                            
                        if not self.running:
                            break
                        continue
                    
                    collection_start = current_time
                    collection_count += 1
                    last_collection = current_time  # Update last collection time
                    
                    logger.info("-" * 60)
                    logger.info(f"Data collection cycle #{collection_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Collect live data
                    wait_time, info = self.tuner.collect_live_data()
                    last_collection = current_time  # Update last collection time

                    logger.info(f"Collected wait time: {wait_time} minutes")
                    logger.info(f"Border crossing status: {info.get('port_status', 'unknown')}")
                    logger.info(f"Last border update time: {info.get('update_time', 'unknown')}")
                    logger.info(f"Next collection scheduled for: {last_collection + timedelta(seconds=self.collection_interval)}")
                    
                    # Check if we have enough data for retraining
                    if self.tuner.should_retrain():
                        logger.info(f"Current data points: {len(self.tuner.live_data)} / {self.tuner.retraining_threshold} required for retraining")
                        logger.info("Retraining model with new data...")
                        try:
                            metrics, new_model = self.tuner.fine_tune_model()
                            if new_model:
                                self.model = new_model
                                logger.info("Model retrained successfully")
                                logger.info(f"New metrics: {metrics}")
                        except Exception as e:
                            logger.error(f"Error in model fine-tuning: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            logger.info(f"Will retry in {self.collection_interval//60} minutes...")
                    else:
                        logger.info(f"Current data points: {len(self.tuner.live_data)} / {self.tuner.retraining_threshold} required for retraining")
                        logger.info(f"Next collection scheduled for: {last_collection + timedelta(seconds=self.collection_interval)}")
                except Exception as e:
                    logger.error(f"Error in data collection cycle: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.info(f"Will retry in {self.collection_interval//60} minutes...")
                    time.sleep(self.collection_interval)  # Wait before retrying
        except Exception as e:
            logger.error(f"Fatal error in daemon: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Final cleanup
            if hasattr(self, '_shutdown_started'):
                shutdown_duration = time.time() - self._shutdown_start_time
                logger.info(f"Shutdown completed after {shutdown_duration:.1f} seconds")
            logger.info("Daemon process terminated")

def main():
    """Main entry point for the daemon process."""
    try:
        # Create and load the pre-trained model
        model = BorderProphetModel()
        
        try:
            # Try to load the existing model
            model.load_model()
            logger.info("Successfully loaded pre-trained model")
        except FileNotFoundError:
            logger.error("No pre-trained model found. Please train a model first using run_training.py")
            sys.exit(1)
        
        # Create and run the daemon
        daemon = LiveDataDaemon(model)
        daemon.run()
        
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
