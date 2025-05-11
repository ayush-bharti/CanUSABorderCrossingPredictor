"""Flask web application for border crossing prediction."""
from typing import Dict, Optional
import os
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from border_predictor.models.prophet_model import BorderProphetModel
from border_predictor.data.data_loader import load_processed_data
from border_predictor.utils.helpers import format_time
from border_predictor.config.config import WEB_CONFIG, MODEL_CONFIG, BORDER_CONFIG


app = Flask(__name__, 
            template_folder=Path(__file__).parent.parent.parent / "templates",
            static_folder=Path(__file__).parent.parent.parent / "static")

# Global model instance
model = None


def load_model() -> BorderProphetModel:
    """
    Load the Prophet model.
    
    Returns:
        Loaded BorderProphetModel instance.
    """
    global model
    
    if model is None:
        model = BorderProphetModel()
        try:
            model.load_model()
        except FileNotFoundError:
            # Return uninitialized model if file not found
            pass
    
    return model


def get_model_status() -> Dict:
    """
    Get the status of the model.
    
    Returns:
        Dictionary with model status information.
    """
    global model
    
    if model is None:
        model = load_model()
    
    status = {
        "model_loaded": model.model is not None,
        "model_path": str(MODEL_CONFIG["model_path"]),
        "crossing_name": BORDER_CONFIG["crossing_name"]
    }
    
    return status


@app.route('/')
def index():
    """Render the home page."""
    # Get model status
    status = get_model_status()
    
    # Generate date options for the next week
    now = datetime.now()
    date_options = []
    for i in range(7):
        date = now + timedelta(days=i)
        date_options.append(date.strftime('%Y-%m-%d'))
    
    # Generate hour options
    hour_options = [f"{h:02d}:00" for h in range(24)]
    
    return render_template(
        'index.html',
        status=status,
        date_options=date_options,
        hour_options=hour_options
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    # Get form data
    date = request.form.get('date')
    time = request.form.get('time')
    current_wait = request.form.get('current_wait')
    
    # Validate inputs
    if not date or not time:
        return jsonify({
            'error': 'Date and time are required'
        }), 400
    
    # Parse current wait time
    current_wait_time = None
    if current_wait:
        try:
            current_wait_time = float(current_wait)
        except ValueError:
            return jsonify({
                'error': 'Current wait time must be a number'
            }), 400
    
    # Combine date and time
    target_date = f"{date} {time}"
    
    try:
        # Load model
        model = load_model()
        
        # Check if model is loaded
        if model.model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Create future dataframe for prediction
        target_date_ts = pd.Timestamp(target_date)
        future_df = pd.DataFrame({'ds': [target_date_ts]})
        
        # Make prediction
        forecast = model.predict(future_df=future_df)
        
        # Extract prediction
        prediction = forecast.iloc[0]
        
        # Format results
        result = {
            'target_date': target_date_ts.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_wait_time': float(prediction['yhat']),
            'predicted_wait_time_formatted': format_time(prediction['yhat']),
            'lower_bound': float(prediction['yhat_lower']),
            'upper_bound': float(prediction['yhat_upper']),
            'crossing_name': BORDER_CONFIG['crossing_name']
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/visualize')
def visualize():
    """Render the visualization page."""
    try:
        # Load processed data
        df = load_processed_data()
        
        # Create visualizations
        # 1. Wait time by hour of day
        df['hour'] = df['ds'].dt.hour
        hourly_avg = df.groupby('hour')['y'].mean().reset_index()
        
        hour_fig = px.bar(
            hourly_avg, 
            x='hour', 
            y='y',
            labels={'hour': 'Hour of Day', 'y': 'Average Wait Time (minutes)'},
            title='Average Border Crossing Wait Time by Hour of Day'
        )
        hour_fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        
        # 2. Wait time by day of week
        df['day_of_week'] = df['ds'].dt.dayofweek
        daily_avg = df.groupby('day_of_week')['y'].mean().reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg['day_name'] = daily_avg['day_of_week'].apply(lambda x: days[x])
        
        day_fig = px.bar(
            daily_avg, 
            x='day_name', 
            y='y',
            labels={'day_name': 'Day of Week', 'y': 'Average Wait Time (minutes)'},
            title='Average Border Crossing Wait Time by Day of Week',
            category_orders={"day_name": days}
        )
        
        # 3. Wait time by month
        df['month'] = df['ds'].dt.month
        monthly_avg = df.groupby('month')['y'].mean().reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: months[x-1])
        
        month_fig = px.bar(
            monthly_avg, 
            x='month_name', 
            y='y',
            labels={'month_name': 'Month', 'y': 'Average Wait Time (minutes)'},
            title='Average Border Crossing Wait Time by Month',
            category_orders={"month_name": months}
        )
        
        # Convert figures to JSON
        hour_json = json.dumps(hour_fig.to_dict())
        day_json = json.dumps(day_fig.to_dict())
        month_json = json.dumps(month_fig.to_dict())
        
        return render_template(
            'visualize.html',
            hour_json=hour_json,
            day_json=day_json,
            month_json=month_json
        )
    
    except FileNotFoundError:
        return render_template(
            'visualize.html',
            error='No processed data found. Please train the model first.'
        )
    
    except Exception as e:
        return render_template(
            'visualize.html',
            error=str(e)
        )


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


def main():
    """Run the Flask application."""
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug']
    )


if __name__ == '__main__':
    main()
