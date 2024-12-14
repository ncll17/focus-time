#!/bin/bash

echo "Upgrading pip to the latest version..."
pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }

echo "All dependencies installed successfully."

# Run your Python script
python train.py
