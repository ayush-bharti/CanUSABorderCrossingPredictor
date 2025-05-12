"""Module for fetching and processing live border crossing wait times from CBP API."""
from typing import Dict, List, Optional, Union
import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from border_predictor.config.config import DATA_CONFIG, BORDER_CONFIG


def fetch_cbp_wait_times(url: str = "https://bwt.cbp.gov/api/waittimes") -> List[Dict]:
    """
    Fetch current wait times from CBP API.
    
    Args:
        url: The URL of the CBP API.
        
    Returns:
        JSON data with wait time data.
        
    Raises:
        requests.RequestException: If the request fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from CBP API: {e}")
        raise


def parse_wait_times_json(json_data: List[Dict], crossing_name: Optional[str] = None) -> pd.DataFrame:
    """
    Parse JSON data from CBP API and extract wait times.
    
    Args:
        json_data: JSON data from CBP API.
        crossing_name: Optional name of the crossing to filter for.
                      If None, uses the crossing name from BORDER_CONFIG.
    
    Returns:
        DataFrame with parsed wait times.
    """
    if crossing_name is None:
        crossing_name = BORDER_CONFIG["crossing_name"]
    
    # List to store wait time data
    wait_times = []
    
    # Current timestamp for when data was fetched
    timestamp = datetime.now()
    
    # Parse each port entry in the JSON data
    for port in json_data:
        port_crossing_name = port.get("crossing_name", "")
        
        # If no crossing name is specified, or if the specified crossing is found
        if crossing_name is None or (port_crossing_name and crossing_name.lower() in port_crossing_name.lower()):
            # Get passenger vehicle lanes
            vehicle_lanes = port.get("passenger_vehicle_lanes", {})
            if vehicle_lanes:
                # Standard lanes
                standard_lanes = vehicle_lanes.get("standard_lanes", {})
                delay_mins = standard_lanes.get("delay_minutes", "")
                if delay_mins and delay_mins.strip():
                    try:
                        delay_mins = int(delay_mins)
                        wait_times.append({
                            "crossing_name": port_crossing_name,
                            "lane_type": "standard",
                            "wait_time": delay_mins,
                            "timestamp": timestamp,
                            "port_status": port.get("port_status", ""),
                            "update_time": standard_lanes.get("update_time", "")
                        })
                    except ValueError:
                        print(f"Warning: Invalid delay minutes value for {port_crossing_name}: {delay_mins}")
                
                # NEXUS/SENTRI lanes (trusted traveler)
                nexus_lanes = vehicle_lanes.get("NEXUS_SENTRI_lanes", {})
                delay_mins = nexus_lanes.get("delay_minutes", "")
                if delay_mins and delay_mins.strip():
                    try:
                        delay_mins = int(delay_mins)
                        wait_times.append({
                            "crossing_name": port_crossing_name,
                            "lane_type": "nexus",
                            "wait_time": delay_mins,
                            "timestamp": timestamp,
                            "port_status": port.get("port_status", ""),
                            "update_time": nexus_lanes.get("update_time", "")
                        })
                    except ValueError:
                        print(f"Warning: Invalid delay minutes value for {port_crossing_name} NEXUS: {delay_mins}")
    
    # Convert to DataFrame
    df = pd.DataFrame(wait_times)
    
    if df.empty:
        print(f"Warning: No data found for crossing '{crossing_name}'")
    
    return df


def get_current_wait_time(crossing_name: Optional[str] = None, 
                          lane_type: str = "standard",
                          api_url: str = "https://bwt.cbp.gov/api/waittimes") -> int:
    """
    Get the current wait time for a specific border crossing.
    
    Args:
        crossing_name: Name of the border crossing. If None, uses the default from config.
        lane_type: Type of lane ('standard' or 'nexus').
        api_url: URL of the CBP API.
        
    Returns:
        Current wait time in minutes.
        
    Raises:
        ValueError: If no wait time data is found for the specified crossing and lane type.
    """
    if crossing_name is None:
        crossing_name = BORDER_CONFIG["crossing_name"]
    
    json_data = fetch_cbp_wait_times(api_url)
    df = parse_wait_times_json(json_data, crossing_name)
    
    if df.empty:
        raise ValueError(f"No wait time data found for {crossing_name}")
    
    filtered_df = df[df["lane_type"] == lane_type]
    
    if filtered_df.empty:
        raise ValueError(f"No wait time data found for {crossing_name} with lane type {lane_type}")
    
    return filtered_df["wait_time"].iloc[0]


def prepare_live_data_for_model(wait_time: int) -> pd.DataFrame:
    """
    Prepare live data for the Prophet model.
    
    Args:
        wait_time: Current wait time in minutes.
        
    Returns:
        DataFrame with 'ds' and 'y' columns ready for Prophet.
    """
    now = datetime.now()
    
    # Create DataFrame with current timestamp and wait time
    df = pd.DataFrame({
        "ds": [now],
        "y": [wait_time]
    })
    
    # Add features
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['year'] = df['ds'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df


def save_live_data(df: pd.DataFrame, file_path: Optional[Path] = None) -> None:
    """
    Save live data to a CSV file.
    
    Args:
        df: DataFrame with live data.
        file_path: Path to save the data. If None, uses the default path.
    """
    if file_path is None:
        file_path = DATA_CONFIG["data_dir"] / "live_wait_times.csv"
    
    # Check if file exists and append if it does
    if file_path.exists():
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df])
        combined_df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)
