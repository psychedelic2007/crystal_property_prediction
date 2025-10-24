#!/bin/bash

# Backend startup script for Crystal Structure Prediction API

echo "Starting Crystal Structure Prediction Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the FastAPI server
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload



