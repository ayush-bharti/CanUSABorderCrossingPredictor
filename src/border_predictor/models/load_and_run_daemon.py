"""Script to load trained model and run the live data daemon."""
import pickle
import logging
from pathlib import Path

from border_predictor.config.config import MODEL_CONFIG
from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.models.live_data_daemon import LiveDataDaemon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_model():
    """Load the latest trained model from disk."""
    model_path = Path(MODEL_CONFIG["model_path"])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the trained model file
    with open(model_path, 'rb') as f:
        prophet_model = pickle.load(f)
    
    # Create a BorderProphetModel instance and attach the loaded model
    model = BorderProphetModel()
    model.model = prophet_model
    
    logger.info(f"Loaded trained model from {model_path}")
    return model

def main():
    """Main entry point to load the model and run the daemon."""
    try:
        # Load the trained model
        logger.info("Loading trained Prophet model...")
        model = load_trained_model()
        
        # Create and run the daemon
        logger.info("Initializing live data daemon...")
        daemon = LiveDataDaemon(model)
        
        # Add signal handlers
        import signal
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}. Shutting down daemon...")
            daemon.running = False
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        
        # Run the daemon
        logger.info("Starting live data collection daemon...")
        logger.info("Press Ctrl+C to stop the daemon")
        daemon.run()
        
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
    except Exception as e:
        logger.error(f"Failed to start daemon: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
