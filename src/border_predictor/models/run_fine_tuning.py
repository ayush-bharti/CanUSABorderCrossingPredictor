"""Script to run model fine-tuning on collected live data."""
import pickle
from pathlib import Path
import logging

from border_predictor.config.config import MODEL_CONFIG
from border_predictor.models.live_model_tuner import LiveModelTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_fine_tuning():
    """Run fine-tuning on the existing model using collected live data."""
    try:
        # Load the existing model
        model_path = Path(MODEL_CONFIG["model_path"])
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Create tuner instance
        tuner = LiveModelTuner(model)
        
        # Run fine-tuning cycle
        logger.info("Starting fine-tuning process...")
        success, metrics = tuner.run_fine_tuning_cycle()
        
        if success:
            logger.info("Fine-tuning successful!")
            logger.info(f"New model metrics: {metrics}")
            
            # Save the updated model
            with open(model_path, 'wb') as f:
                pickle.dump(tuner.model, f)
            logger.info(f"Updated model saved to {model_path}")
        else:
            if metrics:
                logger.warning("Fine-tuning failed performance validation")
                logger.info(f"Metrics from attempted fine-tuning: {metrics}")
            else:
                logger.warning("Fine-tuning skipped - insufficient data or other error")
                
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    run_fine_tuning()
