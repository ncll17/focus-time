#!/bin/bash

# Install PDM if not available
if ! command -v pdm &> /dev/null; then
    pip install pdm
fi

# Install dependencies from pdm.lock
pdm install --lock

# Run your Python script
python train.py