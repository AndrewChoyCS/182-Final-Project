#!/bin/bash
# Quick start script for CS182 Final Project

echo "CS182 Final Project - Quick Start"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

echo ""
echo "Setup complete!"
echo ""
echo "To train a model, run:"
echo "  python train.py --config config.yaml"
echo ""
echo "To evaluate a model, run:"
echo "  python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth"
echo ""

