"""Module for fetching and processing live border crossing wait times from CBP API."""
from typing import Dict, List, Optional, Union, Tuple
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


def find_crossing(
    json_data: List[Dict], 
    port_name: Optional[str] = None,
    crossing_name: Optional[str] = None
) -> Optional[Dict]:
    """
    Find a specific border crossing in the API data.
    
    Args:
        json_data: JSON data from CBP API.
        port_name: Name of the port to search for.
        crossing_name: Name of the crossing to search for.
        
    Returns:
        Dictionary with crossing data if found, None otherwise.
    """
    if port_name is None:
        port_name = BORDER_CONFIG.get("port_name")
        
    if crossing_name is None:
        crossing_name = BORDER_CONFIG.get("crossing_name")
    
    # First try exact match with both port_name and crossing_name
    if port_name and crossing_name:
        for port in json_data:
            if (port.get("port_name") == port_name and 
                port.get("crossing_name") == crossing_name):
                return port
    
    # Try partial match with port_name only
    if port_name:
        matches = []
        for port in json_data:
            if port_name.lower() in port.get("port_name", "").lower():
                matches.append(port)
                
        # If only one match is found, return it
        if len(matches) == 1:
            return matches[0]
        
        # If multiple matches but one has the crossing_name, return it
        if crossing_name and len(matches) > 1:
            for port in matches:
                if crossing_name.lower() in port.get("crossing_name", "").lower():
                    return port
    
    # Try partial match with crossing_name only
    if crossing_name:
        matches = []
        for port in json_data:
            if crossing_name.lower() in port.get("crossing_name", "").lower():
                matches.append(port)
                
        # If only one match is found, return it
        if len(matches) == 1:
            return matches[0]
    
    return None


def parse_wait_times_json(
    json_data: List[Dict], 
    port_name: Optional[str] = None,
    crossing_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse JSON data from CBP API and extract wait times.
    
    Args:
        json_data: JSON data from CBP API.
        port_name: Name of the port to filter for.
        crossing_name: Name of the crossing to filter for.
    
    Returns:
        DataFrame with parsed wait times.
    """
    if port_name is None:
        port_name = BORDER_CONFIG.get("port_name")
        
    if crossing_name is None:
        crossing_name = BORDER_CONFIG.get("crossing_name")
    
    # List to store wait time data
    wait_times = []
    
    # Current timestamp for when data was fetched
    timestamp = datetime.now()
    
    # Find the specific crossing if port_name and crossing_name are provided
    if port_name or crossing_name:
        port = find_crossing(json_data, port_name, crossing_name)
        if port:
            # Process this specific port
            port_name_found = port.get("port_name", "")
            crossing_name_found = port.get("crossing_name", "")
            
            # Get passenger vehicle lanes
            vehicle_lanes = port.get("passenger_vehicle_lanes", {})
            if vehicle_lanes:
                # Standard lanes
                std_lanes = vehicle_lanes.get("standard_lanes", {})
                delay_mins = std_lanes.get("delay_minutes", "")
                if delay_mins and delay_mins.strip():
                    try:
                        delay_mins = int(delay_mins)
                        wait_times.append({
                            "port_name": port_name_found,
                            "crossing_name": crossing_name_found,
                            "lane_type": "standard",
                            "wait_time": delay_mins,
                            "timestamp": timestamp,
                            "port_status": port.get("port_status", ""),
                            "update_time": std_lanes.get("update_time", "")
                        })
                    except ValueError:
                        print(f"Warning: Invalid delay minutes value for {port_name_found} {crossing_name_found}: {delay_mins}")
                
                # NEXUS/SENTRI lanes (trusted traveler)
                nexus_lanes = vehicle_lanes.get("NEXUS_SENTRI_lanes", {})
                delay_mins = nexus_lanes.get("delay_minutes", "")
                if delay_mins and delay_mins.strip():
                    try:
                        delay_mins = int(delay_mins)
                        wait_times.append({
                            "port_name": port_name_found,
                            "crossing_name": crossing_name_found,
                            "lane_type": "nexus",
                            "wait_time": delay_mins,
                            "timestamp": timestamp,
                            "port_status": port.get("port_status", ""),
                            "update_time": nexus_lanes.get("update_time", "")
                        })
                    except ValueError:
                        print(f"Warning: Invalid delay minutes value for {port_name_found} {crossing_name_found} NEXUS: {delay_mins}")
    else:
        # Process all ports if no specific port or crossing is requested
        for port in json_data:
            port_name_found = port.get("port_name", "")
            crossing_name_found = port.get("crossing_name", "")
            
            # Get passenger vehicle lanes
            vehicle_lanes = port.get("passenger_vehicle_lanes", {})
            if vehicle_lanes:
                # Standard lanes
                std_lanes = vehicle_lanes.get("standard_lanes", {})
                delay_mins = std_lanes.get("delay_minutes", "")
                if delay_mins and delay_mins.strip():
                    try:
                        delay_mins = int(delay_mins)
                        wait_times.append({
                            "port_name": port_name_found,
                            "crossing_name": crossing_name_found,
                            "lane_type": "standard",
                            "wait_time": delay_mins,
                            "timestamp": timestamp,
                            "port_status": port.get("port_status", ""),
                            "update_time": std_lanes.get("update_time", "")
                        })
                    except ValueError:
                        print(f"Warning: Invalid delay minutes value for {port_name_found} {crossing_name_found}: {delay_mins}")
                
                # NEXUS/SENTRI lanes (trusted traveler)
                nexus_lanes = vehicle_lanes.get("NEXUS_SENTRI_lanes", {})
                delay_mins = nexus_lanes.get("delay_minutes", "")
                if delay_mins and delay_mins.strip():
                    try:
                        delay_mins = int(delay_mins)
                        wait_times.append({
                            "port_name": port_name_found,
                            "crossing_name": crossing_name_found,
                            "lane_type": "nexus",
                            "wait_time": delay_mins,
                            "timestamp": timestamp,
                            "port_status": port.get("port_status", ""),
                            "update_time": nexus_lanes.get("update_time", "")
                        })
                    except ValueError:
                        print(f"Warning: Invalid delay minutes value for {port_name_found} {crossing_name_found} NEXUS: {delay_mins}")
    
    # Convert to DataFrame
    df = pd.DataFrame(wait_times)
    
    if df.empty:
        print(f"Warning: No data found for port '{port_name}' crossing '{crossing_name}'")
    
    return df


def get_current_wait_time(
    port_name: Optional[str] = None, 
    crossing_name: Optional[str] = None,
    lane_type: str = "standard",
    api_url: str = "https://bwt.cbp.gov/api/waittimes"
) -> Tuple[int, dict]:
    """
    Get the current wait time for a specific border crossing.
    
    Args:
        port_name: Name of the port. If None, uses the default from config.
        crossing_name: Name of the crossing. If None, uses the default from config.
        lane_type: Type of lane ('standard' or 'nexus').
        api_url: URL of the CBP API.
        
    Returns:
        Tuple of (wait_time, crossing_info) where:
          - wait_time is the current wait time in minutes
          - crossing_info is a dictionary with additional information about the crossing
        
    Raises:
        ValueError: If no wait time data is found for the specified crossing and lane type.
    """
    if port_name is None:
        port_name = BORDER_CONFIG.get("port_name")
        
    if crossing_name is None:
        crossing_name = BORDER_CONFIG.get("crossing_name")
    
    json_data = fetch_cbp_wait_times(api_url)
    df = parse_wait_times_json(json_data, port_name, crossing_name)
    
    if df.empty:
        raise ValueError(f"No wait time data found for port '{port_name}' crossing '{crossing_name}'")
    
    filtered_df = df[df["lane_type"] == lane_type]
    
    if filtered_df.empty:
        raise ValueError(f"No wait time data found for port '{port_name}' crossing '{crossing_name}' with lane type {lane_type}")
    
    # Get the first matching wait time
    wait_time = filtered_df["wait_time"].iloc[0]
    
    # Create a dictionary with crossing info
    crossing_info = {
        "port_name": filtered_df["port_name"].iloc[0],
        "crossing_name": filtered_df["crossing_name"].iloc[0],
        "port_status": filtered_df["port_status"].iloc[0],
        "update_time": filtered_df["update_time"].iloc[0],
        "timestamp": filtered_df["timestamp"].iloc[0]
    }
    
    return wait_time, crossing_info


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
