"""Test script for the updated live data loader module."""
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from border_predictor.data.live_data_loader_updated import (
    fetch_cbp_wait_times,
    parse_wait_times_json,
    get_current_wait_time,
    prepare_live_data_for_model,
    save_live_data
)
from border_predictor.config.config import BORDER_CONFIG, LIVE_DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_updated_live_data_fetching():
    """Test the updated live data loader implementation."""
    try:
        # Get configuration values
        port_name = BORDER_CONFIG.get("port_name")
        crossing_name = BORDER_CONFIG.get("crossing_name")
        lane_type = BORDER_CONFIG.get("lane_type")
        
        logger.info(f"Testing updated live data loader for {port_name} - {crossing_name}")
        
        # Fetch data from CBP API
        logger.info(f"Fetching data from CBP API: {LIVE_DATA_CONFIG['api_url']}")
        json_data = fetch_cbp_wait_times(LIVE_DATA_CONFIG['api_url'])
        logger.info(f"Retrieved {len(json_data)} entries from CBP API")
        
        # Parse wait times for specific crossing
        logger.info(f"Parsing wait times for {port_name} - {crossing_name}")
        crossing_df = parse_wait_times_json(json_data, port_name, crossing_name)
        
        if not crossing_df.empty:
            logger.info(f"Successfully found data for {port_name} - {crossing_name}")
            logger.info(f"Available lane types: {', '.join(crossing_df['lane_type'].unique())}")
            
            # Display wait times for each lane type
            for _, row in crossing_df.iterrows():
                logger.info(f"  Lane type: {row['lane_type']}, Wait time: {row['wait_time']} minutes")
                logger.info(f"  Last updated: {row['update_time']}")
            
            # Save data to CSV
            output_path = Path(f"data/{port_name}_{crossing_name.replace(' ', '_')}_wait_times.csv")
            crossing_df.to_csv(output_path, index=False)
            logger.info(f"Saved crossing data to {output_path}")
            
            # Get current wait time
            try:
                wait_time, crossing_info = get_current_wait_time(
                    port_name=port_name,
                    crossing_name=crossing_name,
                    lane_type=lane_type
                )
                
                logger.info(f"Current wait time for {lane_type} lanes: {wait_time} minutes")
                logger.info(f"Crossing info: {crossing_info}")
                
                # Prepare and save live data for model
                live_df = prepare_live_data_for_model(wait_time)
                logger.info(f"Prepared live data with features: {', '.join(live_df.columns)}")
                
                save_live_data(live_df)
                logger.info("Live data saved successfully")
                
                return wait_time, crossing_info
                
            except ValueError as e:
                logger.error(f"Error getting current wait time: {e}")
                return None, None
        else:
            logger.error(f"No data found for {port_name} - {crossing_name}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in test_updated_live_data_fetching: {e}")
        return None, None


if __name__ == "__main__":
    logger.info("=== Starting test of updated live data loader ===")
    wait_time, crossing_info = test_updated_live_data_fetching()
    
    if wait_time is not None:
        logger.info("=== Test completed successfully ===")
        logger.info(f"Current wait time at {crossing_info['port_name']} - {crossing_info['crossing_name']}: {wait_time} minutes")
    else:
        logger.error("=== Test failed ===")
