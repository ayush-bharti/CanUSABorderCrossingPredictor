"""Script to list all available border crossings from the CBP API."""
import requests
import pandas as pd
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_all_crossings(api_url="https://bwt.cbp.gov/api/waittimes"):
    """Fetch and list all border crossings from the CBP API."""
    try:
        # Fetch the data
        logger.info(f"Fetching data from {api_url}")
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        # Parse the JSON
        crossings_data = response.json()
        logger.info(f"Retrieved data for {len(crossings_data)} border crossing entries")
        
        # Extract relevant information
        crossings_info = []
        for crossing in crossings_data:
            info = {
                "port_name": crossing.get("port_name", ""),
                "crossing_name": crossing.get("crossing_name", ""),
                "border": crossing.get("border", ""),
                "port_status": crossing.get("port_status", ""),
                "hours": crossing.get("hours", "")
            }
            
            # Check for vehicle lanes
            vehicle_lanes = crossing.get("passenger_vehicle_lanes", {})
            if vehicle_lanes:
                # Standard lanes
                std_lanes = vehicle_lanes.get("standard_lanes", {})
                if std_lanes:
                    info["std_wait_min"] = std_lanes.get("delay_minutes", "")
                    info["std_update_time"] = std_lanes.get("update_time", "")
                
                # NEXUS/SENTRI lanes
                nexus_lanes = vehicle_lanes.get("NEXUS_SENTRI_lanes", {})
                if nexus_lanes:
                    info["nexus_wait_min"] = nexus_lanes.get("delay_minutes", "")
                    info["nexus_update_time"] = nexus_lanes.get("update_time", "")
            
            crossings_info.append(info)
        
        # Convert to DataFrame
        df = pd.DataFrame(crossings_info)
        
        # Save to CSV
        output_path = Path("data/all_border_crossings_detailed.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed crossing information to {output_path}")
        
        # Display all crossings with "Blaine" in the name
        logger.info("Searching for 'Blaine' related crossings:")
        blaine_df = df[
            (df["port_name"].str.contains("Blaine", case=False, na=False)) | 
            (df["crossing_name"].str.contains("Blaine", case=False, na=False))
        ]
        
        if not blaine_df.empty:
            logger.info(f"Found {len(blaine_df)} Blaine related crossings:")
            for idx, row in blaine_df.iterrows():
                logger.info(f"  {row['port_name']} - {row['crossing_name']} ({row['border']})")
                if row.get("std_wait_min"):
                    logger.info(f"    Standard lanes wait: {row['std_wait_min']} min")
                if row.get("nexus_wait_min"):
                    logger.info(f"    NEXUS lanes wait: {row['nexus_wait_min']} min")
        else:
            logger.info("No crossings containing 'Blaine' found")
            
        # Display all border crossings
        logger.info("\nAll available border crossings:")
        for border_type in df["border"].unique():
            border_df = df[df["border"] == border_type]
            logger.info(f"\n{border_type} ({len(border_df)} crossings):")
            
            # Group by port name
            port_groups = border_df.groupby("port_name")
            for port_name, port_df in port_groups:
                logger.info(f"  {port_name}:")
                for _, row in port_df.iterrows():
                    crossing_info = f"    {row['crossing_name']}" if row['crossing_name'] else "    (Main crossing)"
                    logger.info(crossing_info)
        
        # Search for "Peace" or "Arch" in case it's listed differently
        logger.info("\nSearching for 'Peace' or 'Arch' related crossings:")
        peace_arch_df = df[
            (df["port_name"].str.contains("Peace", case=False, na=False)) | 
            (df["crossing_name"].str.contains("Peace", case=False, na=False)) |
            (df["port_name"].str.contains("Arch", case=False, na=False)) | 
            (df["crossing_name"].str.contains("Arch", case=False, na=False))
        ]
        
        if not peace_arch_df.empty:
            logger.info(f"Found {len(peace_arch_df)} Peace/Arch related crossings:")
            for idx, row in peace_arch_df.iterrows():
                logger.info(f"  {row['port_name']} - {row['crossing_name']} ({row['border']})")
                if row.get("std_wait_min"):
                    logger.info(f"    Standard lanes wait: {row['std_wait_min']} min")
        else:
            logger.info("No crossings containing 'Peace' or 'Arch' found")
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching border crossings: {e}")
        return None

if __name__ == "__main__":
    fetch_all_crossings()
