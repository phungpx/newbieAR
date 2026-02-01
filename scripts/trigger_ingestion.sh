#!/bin/bash

# Script to trigger the vectordb_ingestion DAG
# Usage: ./scripts/trigger_ingestion.sh [file_path]

set -e

# Default file path if not provided
DEFAULT_FILE_PATH="${PWD}/data/papers/files/docling.pdf"
FILE_PATH="${1:-$DEFAULT_FILE_PATH}"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "❌ Error: File not found: $FILE_PATH"
    echo ""
    echo "Usage: $0 [file_path]"
    echo "Example: $0 /path/to/document.pdf"
    exit 1
fi

# Get absolute path
FILE_PATH=$(realpath "$FILE_PATH")

echo "🚀 Triggering vectordb_ingestion DAG..."
echo "📄 File: $FILE_PATH"
echo ""

# Trigger the DAG with configuration
airflow dags trigger vectordb_ingestion \
  --conf "{\"file_path\": \"$FILE_PATH\"}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DAG triggered successfully!"
    echo "📊 Check the Airflow UI to monitor the run."
else
    echo ""
    echo "❌ Failed to trigger DAG. Check Airflow connection and permissions."
    exit 1
fi
