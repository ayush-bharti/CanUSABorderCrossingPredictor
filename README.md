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

### Cascade Gateway History

`border_predictor.cascade` downloads 5-minute delay, volume and data-availability readings from the
[Cascade Gateway Border Data API](https://www.cascadegatewaydata.com/Api) for Peace Arch and Pacific
Highway, both directions, cars and NEXUS. It needs `CASCADE_API_KEY` in the environment or `.env`.
Files land in `data/cascade/<lane_id>/YYYY-MM.csv`, which is gitignored; months already downloaded
are skipped, so the command can be re-run to resume or to add new months.

```bash
python -m border_predictor.cascade                       # 2007-01 through last full month
python -m border_predictor.cascade --start 2026-09 --end 2026-09
```

### Baseline Predictor

`border_predictor.baseline` predicts a lane's mean delay for an hour from a typical (month, weekday,
hour) profile fitted on ordinary days since 2023 (after the COVID border closure), scaled by multipliers
fitted for Canada/BC and US/WA holidays and the two days either side of them. When a live reading is
available, its gap from the profile is carried forward and fades out exponentially.

How the profile is built, how much of each holiday effect to keep, and how fast the live reading fades
are chosen per lane on a validation year. `train_baseline` does that tuning, refits on all data since
2023 and saves `models/baseline_lane<id>.json` (gitignored, since it is derived from Cascade data).
`predict` loads that model and anchors it on Cascade's current reading for the lane.

```bash
python -m border_predictor.train_baseline                # all lanes; --lanes 4 5 for a subset
python -m border_predictor.predict                       # Peace Arch South Cars, next 12 hours
python -m border_predictor.predict --lane 5 --at "2026-12-24 15:00" --hours 6 --no-live
```

### Evaluation

`border_predictor.evaluate` replays history hour by hour. At each hour the predictor sees only the
live reading available then (mean of the last three 5-minute readings) and predicts the mean delay of
an hour 0 to 168 hours ahead. Settings are tuned on Oct 2024 - Sep 2025, the model is refit on
Jan 2023 - Sep 2025, and it is scored once on Oct 2025 - Sep 2026. It reports mean absolute error,
error on hours over 30 minutes, and how many of those hours were flagged (recall) and how many flags
were right (precision), next to two references: the last reading and the profile alone. Periods are
in `EVALUATION_CONFIG`.

```bash
python -m border_predictor.evaluate --lanes 4
```

### CBP Collector

`border_predictor.collector` records the wait times CBP posts for Peace Arch, Pacific Highway, Lynden
and Sumas into `data/collected/YYYY-MM.csv`, keyed on CBP's own update time so repeated polling never
creates duplicates. It is a record of what travellers are shown, not model input: CBP posts a floor of
about 5 minutes and under-reports long waits compared with Cascade's measured delay.
`.github/workflows/collect.yml` runs it once an hour (20 minutes past) on GitHub Actions.

```bash
python -m border_predictor.collector
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
