#!/bin/sh

# Run the bayesian model for each location:
poetry run python main.py --location "Calgary"
poetry run python main.py --location "Edmonton"
poetry run python main.py --location "Lethbridge"
poetry run python main.py --location "Red Deer"
poetry run python main.py --location "Medicine Hat"
poetry run python main.py --location "Fort McMurray"
poetry run python main.py --location "Grande Prairie"
poetry run python main.py --location "Banff"
poetry run python main.py --location "Jasper"
poetry run python main.py --location "Fort Saskatchewan"