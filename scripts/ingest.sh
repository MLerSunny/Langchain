#!/bin/bash
# Bash script for ingesting documents
# Usage: ./scripts/ingest.sh [--rebuild]

# Default values
rebuild=false
data_dir="data/raw"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rebuild)
      rebuild=true
      ;;
    --data-dir)
      data_dir="$2"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

echo "Starting document ingestion process..."

# Ensure the data directory exists
if [ ! -d "$data_dir" ]; then
    echo "Error: Data directory $data_dir does not exist!"
    exit 1
fi

# Build the command
ingest_command="python ingest/ingest.py --source-dir \"$data_dir\""

# Add rebuild flag if specified
if [ "$rebuild" = true ]; then
    ingest_command+=" --rebuild"
fi

# Run the ingestion script
echo "Running: $ingest_command"
eval $ingest_command

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Ingestion completed successfully!"
else
    echo "Ingestion failed with exit code $?"
    exit 1
fi 