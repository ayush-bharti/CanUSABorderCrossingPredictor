# US-Canada Border Crossing Predictor

Predicts the delay at the Peace Arch and Pacific Highway crossings between Vancouver and Seattle, in both
directions, for cars and NEXUS, from now up to a week ahead.

## Project Overview

- Data: 5-minute measured delay and volume per lane from the Cascade Gateway Border Data Warehouse,
  used both for training (since 2023) and as the live input
- Models per lane: a typical-delay profile by month, weekday and hour with Canadian and US holiday
  adjustments, and LightGBM models that correct it from the latest readings
- Web app comparing both crossings now, over the next 12 hours, and for a planned crossing time

## Project Structure

```
├── src/border_predictor/
│   ├── cascade.py         # Cascade Gateway download and live readings
│   ├── baseline.py        # Typical-delay profile with holiday and live adjustments
│   ├── train_baseline.py  # Tune and save the baseline per lane
│   ├── gbm.py             # LightGBM features, training, evaluation and forecasts
│   ├── evaluate.py        # Hour-by-hour backtest
│   ├── predict.py         # Command-line predictions
│   ├── service.py         # Model loading and cached live data for the web app
│   ├── app.py             # Flask web app and JSON API
│   ├── collector.py       # CBP posted wait-time recorder
│   └── config/            # Configuration
├── tests/
├── static/, templates/    # Web interface
└── data/                  # CBP monthly tables; Cascade downloads and collected data (gitignored)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayush-bharti/CanUSABorderCrossingPredictor.git
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

```bash
python -m border_predictor.train_baseline                # all lanes; --lanes 4 5 for a subset
```

### LightGBM Predictor

`border_predictor.gbm` trains LightGBM models per lane for three ranges of hours ahead (up to 6, 7 to 24,
and 25 to 168), each with its own early stopping, since one model stopped on every horizon at once
under-fits the next few hours. Its inputs are what is known at prediction time: the latest readings, the previous three hours of delay
and volume, the same hour a day and a week before the target when already known, calendar and
holiday-distance features for the target hour, and the baseline's own prediction. It learns
corrections to the baseline, chiefly how a building queue keeps building and how long-weekend travel
days differ from the holidays themselves. The other seven lanes' latest reading and last-hour delay are
inputs too, since a queue in one lane often spills into another.

Each band has four models: the median (the prediction), the 10% and 90% quantiles (a range the delay
falls in 8 times out of 10), and, for the car lanes with enough history of it, a classifier for the
chance that the hour averages over 30 minutes. Its alert threshold is picked per band for the best F1
on the validation year. `train` also refreshes the lane's baseline, then saves
`models/gbm_lane<id>_<model>_upto<h>h.txt` and the thresholds in `models/gbm_lane<id>_meta.json`
(gitignored).

```bash
python -m border_predictor.gbm train                     # all lanes; --lanes 4 for one
python -m border_predictor.gbm evaluate --lanes 4        # compare with the baseline on the test year
```

### Predicting

`predict` fetches the last nine days of readings from Cascade and prints LightGBM's prediction for each
hour with its 80% range and spike alert, and the baseline's next to it. With `--model baseline` it uses the baseline anchored on
Cascade's current reading, and with `--no-live` the baseline's historical profile alone.

```bash
python -m border_predictor.predict                       # Peace Arch South Cars, next 12 hours
python -m border_predictor.predict --lane 5 --hours 48
python -m border_predictor.predict --lane 5 --at "2026-12-24 15:00" --hours 6 --no-live
```

### Evaluation

`border_predictor.evaluate` replays history hour by hour. At each hour the predictor sees only the
live reading available then (mean of the last three 5-minute readings) and predicts the mean delay of
an hour 0 to 168 hours ahead. Settings are tuned on Oct 2024 - Sep 2025, the model is refit on
Jan 2023 - Sep 2025, and it is scored once on Oct 2025 - Sep 2026. It reports mean absolute error,
error on hours over 30 minutes, and how many of those hours were flagged (recall) and how many flags
were right (precision), next to two references: the last reading and the profile alone. Periods are
in `EVALUATION_CONFIG`. `gbm evaluate` scores LightGBM on the same rows.

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

### Web Interface

```bash
python run.py                                            # http://127.0.0.1:5000
```

The home page compares both crossings for a chosen direction and lane type: the latest reading, the
next 12 hours, and any hour in the next two weeks. Hours beyond LightGBM's week, or when Cascade cannot
be reached, show the typical delay instead. The visualize page shows each lane's typical delay by
weekday and hour. It needs trained models (`python -m border_predictor.gbm train`) and
`CASCADE_API_KEY`. JSON is at `/api/forecast?direction=south&kind=cars&hours=12` and
`/api/plan?direction=north&kind=nexus&at=2026-12-24%2015:00`.

## Data Sources

- [Cascade Gateway Border Data Warehouse](https://www.cascadegatewaydata.com/) (Whatcom Council of Governments)
- [CBP Border Wait Times](https://bwt.cbp.gov/) posted wait times
- Canadian (BC) and US (WA) holiday calendars from the `holidays` package

## License

[Your chosen license]
