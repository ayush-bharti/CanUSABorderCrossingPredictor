# US-Canada Border Crossing Predictor

A machine learning model to predict wait times at the Blaine Peace Arch border crossing between Vancouver and Seattle.

## Project Overview

This project uses Facebook's Prophet library to predict border crossing wait times based on:
- Historical average wait times by month, day, and time
- Current border wait time
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

[Instructions will be added as the project develops]

## Data Sources

- Historical border crossing wait times
- US and Canadian holiday calendars
- [Additional data sources will be listed here]

## License

[Your chosen license]
