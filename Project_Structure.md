# US-Canada Border Crossing Predictor - Project Structure

This document provides a detailed overview of the project structure, data flow, important workflows, and key components.

## Project Structure

```
CanUSABorderCrossingPredictor/
├── data/                           # Data storage
│   ├── monthly data files          # Monthly wait time data (Jan-Dec)
│   ├── Blaine_Peace_Arch_wait_times.csv  # Combined historical wait times
│   ├── live_wait_times.csv         # Cached live data from API
│   └── model_evaluation.csv        # Model evaluation results
│
├── models/                         # Saved model files
│   └── .gitkeep                    # Directory placeholder
│
├── src/                            # Source code (src-layout)
│   └── border_predictor/           # Main package
│       ├── __init__.py             # Package initialization
│       ├── app.py                  # Flask web application
│       ├── config/                 # Configuration management
│       │   ├── __init__.py
│       │   └── config.py           # Global configuration settings
│       ├── data/                   # Data processing modules
│       │   ├── __init__.py
│       │   ├── data_loader.py      # Historical data loading utilities
│       │   └── live_data_loader.py  # Live data API integration (JSON-based)
│       ├── models/                 # Model implementation
│       │   ├── __init__.py
│       │   ├── prophet_model.py    # Prophet model implementation
│       │   ├── train_model.py      # Model training script
│       │   ├── live_model_tuner.py # Live data fine-tuning system
│       │   └── evaluate_live.py    # Live evaluation script
│       ├── utils/                  # Helper utilities
│       │   ├── __init__.py
│       │   └── helpers.py          # Common helper functions
│       ├── test_future_prediction.py # Future prediction testing
│
├── static/                         # Static web assets
│   ├── css/                        # CSS stylesheets
│   └── js/                         # JavaScript files
│
├── templates/                      # Jinja2 HTML templates
│   ├── base.html                   # Base template with common structure
│   ├── index.html                  # Main prediction interface
│   └── visualize.html              # Visualization dashboard
│
├── tests/                          # Automated tests
│   ├── test_data_processing.py     # Data processing tests
│   ├── test_imports.py             # Import validation tests
│   └── test_model.py               # Model functionality tests
│
├── requirements.txt                # Project dependencies
├── setup.py                        # Package installation script
├── README.md                       # Project documentation
└── Project_Structure.md            # This document
```

## Data Flow

The data flow in the project follows these main steps:

1. **Data Acquisition**
   - Historical data is loaded from CSV files in the `data/` directory
   - Live data is fetched from the CBP (Customs and Border Protection) API

2. **Data Processing**
   - Historical data is processed and converted to Prophet's required format (ds/y columns)
   - Additional features are added (holidays, day of week, etc.)
   - Data is split into training and testing sets

3. **Model Training**
   - The Prophet model is configured with seasonal parameters
   - Model is trained on historical data with logistic growth constraints
   - Model parameters are optimized for better prediction accuracy

4. **Prediction Generation**
   - The trained model generates predictions for future time periods
   - Predictions include wait times and confidence intervals
   - Logistic growth ensures predictions are positive and realistic

5. **Evaluation and Visualization**
   - Model performance is evaluated against test data and live data
   - Visualizations show predictions, trends, and seasonal patterns
   - Performance metrics are calculated and stored

## Important Workflows

### 1. Model Training Workflow

```
data_loader.py → train_model.py → prophet_model.py (fit) → Saved model file
```

Key steps:
- Load and preprocess historical data
- Configure the Prophet model with appropriate parameters
- Train the model with logistic growth constraints 
- Save the trained model to disk

### 2. Prediction Workflow

```
Saved model file → prophet_model.py (predict) → Prediction results
```

Key steps:
- Load the trained model
- Prepare future dates for prediction
- Generate predictions with confidence intervals
- Apply floor (0) and cap constraints to ensure realistic predictions

### 4. Live Evaluation Workflow

```
live_data_loader.py → evaluate_live.py → Evaluation metrics
```

Key steps:
- Fetch current border wait times from the CBP API
- Compare with model predictions for the same time
- Calculate error metrics (MAE, RMSE, etc.)
- Store and visualize evaluation results

### 4. Web Interface Workflow

```
app.py → Flask routes → templates/views → User interaction
```

Key steps:
- User inputs date/time for prediction
- Flask processes request and calls model
- Results are formatted and displayed in web interface
- Visualizations show prediction trends

## Key Files and Components

### 1. Configuration (`config/config.py`)

Contains key configuration parameters:
- `MODEL_CONFIG`: Settings for the Prophet model (seasonality, holidays, etc.)
- `DATA_CONFIG`: Paths to data files and processing parameters
- `BORDER_CONFIG`: Border crossing specific settings

### 2. Prophet Model (`models/prophet_model.py`)

Core prediction model with key methods:
- `fit()`: Trains the model on historical data with growth constraints
- `predict()`: Generates predictions with non-negative floor values
- `evaluate()`: Calculates performance metrics
- `plot_forecast()`: Visualizes predictions and confidence intervals

### 3. Data Loaders (`data/data_loader.py`, `data/live_data_loader.py`)

Handle data management:
- Historical data loading and preprocessing
- API integration for live border wait times
- Feature engineering and data transformation

### 4. Web Application (`app.py`)

Flask-based web interface:
- Prediction form for user inputs
- Interactive visualizations
- Results display with confidence intervals

## Key Variables and Parameters

### Model Configuration

- `growth`: Set to 'logistic' to prevent negative predictions
- `seasonality_mode`: Controls how seasonality is modeled
- `yearly_seasonality`: Annual patterns in border crossings
- `weekly_seasonality`: Weekly patterns (weekday vs weekend)
- `daily_seasonality`: Time-of-day patterns

### Prediction Constraints

- `cap_value`: Maximum possible wait time (based on historical data)
- `floor_value`: Minimum possible wait time (set to 0)

### Evaluation Metrics

- `mae`: Mean Absolute Error in minutes
- `rmse`: Root Mean Square Error
- `mape`: Mean Absolute Percentage Error
- `coverage`: Percentage of actual values within prediction interval

## Future Enhancement Areas

1. **Additional Features**
   - Weather data integration
   - Special events and local factors
   - Traffic pattern analysis

2. **Model Improvements**
   - Advanced seasonal modeling
   - Ensemble methods for better accuracy
   - Online learning for continuous improvement

3. **Operational Integration**
   - Automated data pipeline
   - Regular model retraining
   - Alert system for unusual wait times

4. **User Interface**
   - Mobile-friendly interface
   - Interactive visualization dashboard
   - Travel planning recommendations
