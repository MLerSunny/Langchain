# PowerShell script for ingesting documents
# Usage: .\scripts\ingest.ps1 [--rebuild]

param (
    [switch]$rebuild = $false,
    [string]$dataDir = "data/raw"
)

Write-Host "Starting document ingestion process..."

# Ensure the data directory exists
if (-not (Test-Path $dataDir)) {
    Write-Error "Data directory $dataDir does not exist!"
    exit 1
}

# Build the command
$ingestCommand = "python ingest/ingest.py ""$dataDir"""

# Add rebuild flag if specified
if ($rebuild) {
    $ingestCommand += " --rebuild"
}

# Run the ingestion script
Write-Host "Running: $ingestCommand"
Invoke-Expression $ingestCommand

# Check if the command was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Ingestion completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Ingestion failed with exit code $LASTEXITCODE" -ForegroundColor Red
} 