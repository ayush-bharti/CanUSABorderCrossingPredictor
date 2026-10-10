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

### Collecting Wait Times

`border_predictor.collector` fetches the CBP feed once and appends new readings for the
Cascade Gateway ports (Peace Arch, Pacific Highway, Lynden, Sumas) to `data/collected/YYYY-MM.csv`.
Rows are keyed on CBP's own update time, so running it more often than CBP updates (hourly)
never creates duplicates.

```bash
python -m border_predictor.collector
```

It is meant to run on a schedule from an always-on machine. `.github/workflows/collect.yml`
runs it once an hour (20 minutes past) on GitHub Actions and commits new readings once the repo is pushed
to GitHub. A laptop cron entry works as a fallback but leaves gaps whenever the machine sleeps:

```
20 * * * * cd /path/to/CanUSABorderCrossingPredictor && venv/bin/python -m border_predictor.collector >> collector.log 2>&1
```

### Baseline Predictions

`border_predictor.baseline` predicts from the CBP historical averages for (month, weekday, hour),
scaled up on US/WA and Canada/BC holidays and the days next to them. When a current reading is
available, its gap from the historical average is carried forward and fades out over a few hours.
Tunables live in `BASELINE_CONFIG` in `config/config.py`.

```bash
python -m border_predictor.predict                       # next 12 hours from now, using the live reading
python -m border_predictor.predict --at "2026-12-24 15:00" --hours 6 --no-live
```

### Training and Fine-tuning the Model (Prophet, legacy)

```bash
# Train the initial model using historical data
python -m border_predictor.models.train_model

# Start live data collection and model fine-tuning
python -m border_predictor.models.live_data_daemon

python -m border_predictor.models.live_model_tuner

# Evaluate model performance with live data
python -m border_predictor.models.evaluate_live
```

The live data daemon process:
1. Collects live wait times every 15 minutes
2. Fine-tunes the model when sufficient new data is available (default: 96 new data points)
3. Benchmarks predictions against actual wait times
4. Adjusts hyperparameters based on performance

#### Running the Live Data Daemon

To run the daemon in the foreground:
```bash
python -m border_predictor.models.live_data_daemon
```

To run in the background with logging:
```bash
nohup python -m border_predictor.models.live_data_daemon > daemon.log 2>&1 &
```

To check the daemon's status:
```bash
tail -f daemon.log  # View real-time logs
ps aux | grep live_data_daemon  # Check if daemon is running
```

To stop the daemon:
```bash
pkill -f "python -m border_predictor.models.live_data_daemon"
```

#### Model Fine-tuning Parameters

The fine-tuning process can be configured with these parameters:
- `collection_interval`: Time between data collection (default: 15 minutes)
- `retraining_threshold`: Number of new data points before retraining (default: 96)
- `max_retrain_window`: Maximum historical data window for retraining (default: 336 hours)

These parameters can be adjusted in the `config/model_config.yaml` file.

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
