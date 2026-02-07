#!/bin/bash

echo "Starting newbieAR API..."

# Activate virtual environment if needed
# source .venv/bin/activate

# Run with uvicorn
python -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
