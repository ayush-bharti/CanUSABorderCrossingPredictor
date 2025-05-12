"""Test script for fetching live data from CBP API using JSON format."""
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime

from border_predictor.data.live_data_loader_json import (
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


def test_live_data_fetching() -> Optional[int]:
    """
    Test fetching and processing live data from CBP API.
    
    Returns:
        Wait time (in minutes) if successful, None otherwise.
    """
    try:
        # Fetch JSON data from CBP API
        logger.info(f"Fetching data from CBP API: {LIVE_DATA_CONFIG['api_url']}")
        json_data = fetch_cbp_wait_times(LIVE_DATA_CONFIG['api_url'])
        
        # Parse all crossings 
        logger.info("Parsing all border crossings")
        all_crossings_df = parse_wait_times_json(json_data, crossing_name=None)
        
        # Display available crossings
        if not all_crossings_df.empty:
            unique_crossings = all_crossings_df['crossing_name'].unique()
            logger.info(f"Found {len(unique_crossings)} border crossings:")
            for crossing in unique_crossings:
                logger.info(f"  - {crossing}")
            
            # Save all crossings data to a file for reference
            all_crossings_file = Path("data/all_border_crossings.csv")
            all_crossings_df.to_csv(all_crossings_file, index=False)
            logger.info(f"Saved all crossings data to {all_crossings_file}")
        else:
            logger.warning("No border crossings found in the API response")
        
        # Parse specific crossing (Peace Arch)
        crossing_name = BORDER_CONFIG["crossing_name"]
        logger.info(f"Parsing data for specific crossing: {crossing_name}")
        crossing_df = parse_wait_times_json(json_data, crossing_name=crossing_name)
        
        if not crossing_df.empty:
            logger.info(f"Found data for {crossing_name}:")
            for _, row in crossing_df.iterrows():
                logger.info(f"  Lane type: {row['lane_type']}, Wait time: {row['wait_time']} minutes")
            
            # Save crossing data
            crossing_file = Path(f"data/{crossing_name.replace(' ', '_').lower()}_wait_times.csv")
            crossing_df.to_csv(crossing_file, index=False)
            logger.info(f"Saved crossing data to {crossing_file}")
        else:
            logger.warning(f"No data found for crossing: {crossing_name}")
            if not all_crossings_df.empty:
                logger.info("Available crossings (partial list):")
                sample_crossings = unique_crossings[:min(10, len(unique_crossings))]
                for crossing in sample_crossings:
                    logger.info(f"  - {crossing}")
                
                # Try to find similar crossings
                logger.info("Looking for similar crossings containing 'Blaine'...")
                for crossing in unique_crossings:
                    if "blaine" in crossing.lower():
                        logger.info(f"  - {crossing}")
        
        # Get current wait time for standard lanes
        try:
            lane_type = BORDER_CONFIG.get("lane_type", "standard")
            logger.info(f"Getting current wait time for {crossing_name} ({lane_type} lanes)")
            
            # First try the exact name from config
            try:
                wait_time = get_current_wait_time(
                    crossing_name=crossing_name,
                    lane_type=lane_type,
                    api_url=LIVE_DATA_CONFIG['api_url']
                )
                logger.info(f"Current wait time: {wait_time} minutes")
            except ValueError as e:
                # If that fails, try searching for crossings containing "Blaine"
                logger.warning(f"Error with exact crossing name: {e}")
                logger.info("Trying with 'Blaine' as the crossing name...")
                wait_time = get_current_wait_time(
                    crossing_name="Blaine",
                    lane_type=lane_type,
                    api_url=LIVE_DATA_CONFIG['api_url']
                )
                logger.info(f"Current wait time: {wait_time} minutes")
            
            # Prepare live data for model
            logger.info("Preparing live data for model")
            live_df = prepare_live_data_for_model(wait_time)
            logger.info(f"Live data prepared: {live_df.shape[0]} rows, {live_df.shape[1]} columns")
            
            # Save live data
            logger.info("Saving live data")
            save_live_data(live_df)
            logger.info("Live data saved successfully")
            
            # Return the wait time for reference
            return wait_time
            
        except ValueError as e:
            logger.error(f"Error getting wait time: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error testing live data: {e}")
        return None


if __name__ == "__main__":
    logger.info("Starting live data test with JSON format")
    wait_time = test_live_data_fetching()
    if wait_time is not None:
        logger.info(f"Test completed successfully. Current wait time: {wait_time} minutes")
    else:
        logger.error("Test failed to get wait time")
