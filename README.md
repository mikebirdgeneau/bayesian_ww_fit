# Bayesian Timeseries Analysis

This reponsitory contains code to analyze timeseries data from Alberta COVID19 Waste Water samples. The code is written in Python and uses the 
Tensorflow Probability (tfp) library for Bayesian inference.

## Installation

The code is written for Python 3.10 and was tested with Tensorflow 2.15 on Linux (with NVIDIA GPU).

1. Clone the repository
2. Create a virtual environment as required.
2. Install the required packages using pip: `pip install -r requirements.txt` OR `poetry install --no-root`
3. Run the code from within the virtual environment:

```$ poetry run python main.py --location Calgary```

or 

```
$ source venv/bin/activate`
$ python main.py --location Calgary
```

The `--location` parameter can be run for different locations.