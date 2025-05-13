# US-Canada Border Crossing Predictor

A machine learning model to predict wait times at the Blaine Peace Arch border crossing between Vancouver and Seattle.

## Project Overview

This project uses Facebook's Prophet library to predict border crossing wait times based on:
- Historical average wait times by month, day, and time
- Live border wait times (collected every 15 minutes)
- Continuous model fine-tuning with real-time data
- Holiday information for both US and Canada
- Additional features that may impact border traffic

## Project Structure

```
├── src/
│   └── border_predictor/
│       ├── data/         # Data loading and processing
│       ├── models/       # Prophet model implementation
│       ├── utils/        # Helper functions
│       └── config/       # Configuration files
├── tests/                # Unit tests
├── static/               # Static files for web interface
├── templates/            # HTML templates for web interface
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CanUSABorderCrossingPredictor.git
cd CanUSABorderCrossingPredictor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training and Fine-tuning the Model

```bash
# Train the initial model using historical data
python -m border_predictor.models.train_model

# Start live data collection and model fine-tuning
python -m border_predictor.models.live_model_tuner

# Evaluate model performance with live data
python -m border_predictor.models.evaluate_live
```

The model automatically:
- Collects live wait times every 15 minutes
- Fine-tunes itself when sufficient new data is available
- Benchmarks predictions against actual wait times
- Adjusts hyperparameters based on performance

### Making Predictions

```bash
# Predict wait times for specific future dates and times
python -m border_predictor.test_future_prediction
```

### Visualization and Analysis

The prediction outputs include:
- Wait time prediction in minutes
- 95% confidence intervals for predictions
- Visualizations of predictions with time series graphs

### Web Interface

A Flask-based web interface is available for interactive predictions:

```bash
# Start the web server
python -m border_predictor.app
```

Then navigate to http://localhost:5000 in your browser to access the interface.

## Data Sources

- Historical border crossing wait times
- US and Canadian holiday calendars
- [Additional data sources will be listed here]

## License

[Your chosen license]
