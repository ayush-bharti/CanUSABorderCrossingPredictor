"""
Test basic imports to verify the package structure is working.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all package modules can be imported."""
    print("Testing imports...")
    
    # Test importing the packages
    try:
        from border_predictor.config import config
        print("✓ Successfully imported config")
        
        from border_predictor.data import data_loader
        print("✓ Successfully imported data_loader")
        
        from border_predictor.models import prophet_model
        print("✓ Successfully imported prophet_model")
        
        from border_predictor.utils import helpers
        print("✓ Successfully imported helpers")
        
        from border_predictor import app
        print("✓ Successfully imported app")
        
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
