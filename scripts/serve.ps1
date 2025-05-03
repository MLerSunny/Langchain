# PowerShell script for serving the RAG system with FastAPI
# Usage: .\scripts\serve.ps1 [--model model_name_or_path]

param (
    [string]$model = "deepseek-ai/deepseek-llm-32b-instruct",
    [switch]$useLora = $false,
    [string]$loraPath = "checkpoints/r1_ins_lora/final",
    [int]$port = 8080
)

# Check if lora path exists when useLora is enabled
if ($useLora -and -not (Test-Path $loraPath)) {
    Write-Error "LoRA adapter path '$loraPath' does not exist!"
    exit 1
}

# Start vLLM server in a separate process
$vllmPort = 8000
$vllmCommand = "python -m vllm.entrypoints.openai.api_server " + `
               "--model $model " + `
               "--port $vllmPort " + `
               "--tensor-parallel-size 1 "

# Add LoRA adapter if enabled
if ($useLora) {
    $vllmCommand += "--peft-model $loraPath "
}

# Start vLLM server in a new PowerShell window
$vllmProcess = Start-Process powershell -ArgumentList "-Command $vllmCommand" -PassThru -WindowStyle Normal
Write-Host "Started vLLM server with PID: $($vllmProcess.Id)" -ForegroundColor Green

# Wait for vLLM server to start up
Write-Host "Waiting for vLLM server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Set environment variables
$env:VLLM_HOST = "http://localhost:$vllmPort"
$env:CHROMA_DIR = "chroma_insurance"
$env:COLLECTION_NAME = "insurance_docs"

# Start the FastAPI server
Write-Host "Starting FastAPI server on port $port..." -ForegroundColor Cyan
$apiCommand = "uvicorn serve.main:app --host 0.0.0.0 --port $port --reload"

Write-Host "Running: $apiCommand"
Write-Host "API server is running. Press Ctrl+C to stop."
Invoke-Expression $apiCommand

# Handle cleanup when the script is interrupted
try {
    # This will keep the script running until Ctrl+C
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Clean up: Stop the vLLM process when the script is interrupted
    if (-not $vllmProcess.HasExited) {
        Write-Host "Stopping vLLM server (PID: $($vllmProcess.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $vllmProcess.Id -Force
    }
} 