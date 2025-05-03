# PowerShell script for running tests
# Usage: .\scripts\test.ps1 [--test-type all|ingest|rag|model]

param (
    [string]$testType = "all"
)

# Create tests directory if it doesn't exist
if (-not (Test-Path "tests")) {
    Write-Host "Creating tests directory..." -ForegroundColor Yellow
    New-Item -Path "tests" -ItemType Directory -Force | Out-Null
}

# Ensure __init__.py exists in tests directory
if (-not (Test-Path "tests/__init__.py")) {
    New-Item -Path "tests/__init__.py" -ItemType File -Force | Out-Null
}

Write-Host "Running tests for the Insurance RAG system..." -ForegroundColor Cyan

# Determine which tests to run
$testCommand = "pytest"

switch ($testType.ToLower()) {
    "ingest" {
        $testCommand += " tests/test_ingest.py -v"
        Write-Host "Running ingestion tests only..." -ForegroundColor Yellow
    }
    "rag" {
        $testCommand += " tests/test_retrieval.py -v"
        Write-Host "Running RAG retrieval tests only..." -ForegroundColor Yellow
    }
    "model" {
        $testCommand += " tests/test_model.py -v"
        Write-Host "Running model tests only..." -ForegroundColor Yellow
    }
    "api" {
        $testCommand += " tests/test_api.py -v"
        Write-Host "Running API tests only..." -ForegroundColor Yellow
    }
    default {
        $testCommand += " tests/ -v"
        Write-Host "Running all tests..." -ForegroundColor Yellow
    }
}

# Run the tests
Write-Host "Running: $testCommand"
Invoke-Expression $testCommand

# Check if the command was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "All tests passed successfully!" -ForegroundColor Green
} else {
    Write-Host "Some tests failed with exit code $LASTEXITCODE" -ForegroundColor Red
} 