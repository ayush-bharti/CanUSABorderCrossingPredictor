"""
Entry point for running the Border Crossing Predictor web application.
"""
from src.border_predictor.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
