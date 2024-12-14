#!/bin/bash

# Ensure PDM is installed
pip install pdm

# Export dependencies to requirements.txt
pdm export --without-hashes --format requirements.txt --output requirements.txt

# Install dependencies
pip install -r requirements.txt

# Run your Python script
python train.py
