#!/bin/bash

# Install PDM if not available
if ! command -v pdm &> /dev/null; then
    pip install --user pdm
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure PDM is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies from pdm.lock
pdm install

# Run your Python script
python train.py
