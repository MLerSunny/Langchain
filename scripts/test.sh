#!/bin/bash
# Bash script for running tests
# Usage: ./scripts/test.sh [--test-type all|ingest|rag|model]

# Default values
test_type="all"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --test-type)
      test_type="$2"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

# Create tests directory if it doesn't exist
if [ ! -d "tests" ]; then
    echo "Creating tests directory..."
    mkdir -p "tests"
fi

# Ensure __init__.py exists in tests directory
if [ ! -f "tests/__init__.py" ]; then
    touch "tests/__init__.py"
fi

echo "Running tests for the Insurance RAG system..."

# Determine which tests to run
test_command="pytest"

case "${test_type,,}" in
    "ingest")
        test_command+=" tests/test_ingest.py -v"
        echo "Running ingestion tests only..."
        ;;
    "rag")
        test_command+=" tests/test_retrieval.py -v"
        echo "Running RAG retrieval tests only..."
        ;;
    "model")
        test_command+=" tests/test_model.py -v"
        echo "Running model tests only..."
        ;;
    "api")
        test_command+=" tests/test_api.py -v"
        echo "Running API tests only..."
        ;;
    *)
        test_command+=" tests/ -v"
        echo "Running all tests..."
        ;;
esac

# Run the tests
echo "Running: $test_command"
eval $test_command

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
else
    echo "Some tests failed with exit code $?"
    exit 1
fi 