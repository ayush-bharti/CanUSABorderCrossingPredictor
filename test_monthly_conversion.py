#!/usr/bin/env python3
"""
Test script to convert monthly wait time data to Prophet format.
This script demonstrates how to use the convert_monthly_data_to_prophet_format function.
"""
import os
from pathlib import Path
import pandas as pd

from src.border_predictor.data.data_loader import convert_monthly_data_to_prophet_format
from src.border_predictor.config.config import DATA_CONFIG

def main():
    # Get the data directory
    data_dir = DATA_CONFIG["data_dir"]
    
    # Define output path for the converted data
    output_path = data_dir / "monthly_converted_data.csv"
    
    print(f"Converting monthly data files from {data_dir}")
    
    # Convert monthly data to Prophet format
    try:
        df = convert_monthly_data_to_prophet_format(
            data_dir=data_dir,
            reference_year=2023,  # Use any reference year
            save_path=output_path
        )
        
        # Display summary of the converted data
        print("\nConversion successful!")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
        print(f"Wait time range: {df['y'].min()} to {df['y'].max()} minutes")
        
        # Display sample of the data
        print("\nSample of converted data:")
        print(df.head(10))
        
        # Display counts by month
        print("\nRecords by month:")
        month_counts = df.groupby(df['ds'].dt.month).size()
        for month, count in month_counts.items():
            print(f"Month {month}: {count} records")
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main()
