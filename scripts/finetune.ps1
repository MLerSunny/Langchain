# PowerShell script for fine-tuning DeepSeek-LLM
# Usage: .\scripts\finetune.ps1

param (
    [string]$modelName = "deepseek-r1:32b",
    [string]$configFile = "core/deepspeed_zero3.json",
    [string]$outputDir = "checkpoints/r1_ins_lora",
    [int]$trainEpochs = 3,
    [int]$batchSize = 1,
    [int]$gradAccumSteps = 8,
    [switch]$skipDeepSpeed = $false,
    [switch]$useOllama = $false,
    [string]$Target
)

# Check if using Ollama directly
if ($useOllama -or $Target -eq "ollama") {
    Write-Host "Using Ollama with model: $modelName" -ForegroundColor Cyan
    Write-Host "Starting Ollama inference..."
    python inference.py
    exit 0
}

Write-Host "Starting LLM fine-tuning with QLoRA..." -ForegroundColor Cyan

# Check if DeepSpeed is installed
$deepspeedInstalled = $false
try {
    $deepspeedVersion = Invoke-Expression "deepspeed --version"
    $deepspeedInstalled = $true
    Write-Host "Found DeepSpeed version: $deepspeedVersion" -ForegroundColor Green
} catch {
    if (-not $skipDeepSpeed) {
        Write-Host "DeepSpeed is not installed or not in your PATH!" -ForegroundColor Yellow
        Write-Host "You can install it with: pip install deepspeed" -ForegroundColor Yellow
        Write-Host "Or run this script with -skipDeepSpeed to use non-DeepSpeed training" -ForegroundColor Yellow
        Write-Host "Example: .\scripts\finetune.ps1 -skipDeepSpeed" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "DeepSpeed not found. Using standard training instead." -ForegroundColor Yellow
}

# Ensure the config file exists if using DeepSpeed
if (-not $skipDeepSpeed -and -not (Test-Path $configFile)) {
    Write-Error "DeepSpeed config file $configFile does not exist!"
    exit 1
}

# Ensure the dataset directory exists
if (-not (Test-Path "finetune/dataset")) {
    Write-Error "Dataset directory 'finetune/dataset' does not exist or is empty!"
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $outputDir)) {
    New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
    Write-Host "Created output directory: $outputDir"
}

# Set environment variables for training
$env:CUDA_VISIBLE_DEVICES = "0"
$env:TOKENIZERS_PARALLELISM = "false"

# Build the command
if (-not $skipDeepSpeed -and $deepspeedInstalled) {
    # DeepSpeed command
    $command = "deepspeed --num_gpus=1 finetune/trainer.py " + `
                        "--deepspeed $configFile " + `
                        "--model_name $modelName " + `
                        "--dataset_dir finetune/dataset " + `
                        "--output_dir $outputDir " + `
                        "--num_epochs $trainEpochs " + `
                        "--batch_size $batchSize " + `
                        "--gradient_accumulation_steps $gradAccumSteps " + `
                        "--learning_rate 2e-4 " + `
                        "--logging_steps 10 " + `
                        "--save_steps 100 " + `
                        "--warmup_steps 50 " + `
                        "--max_seq_length 2048 " + `
                        "--lora_r 64 " + `
                        "--lora_alpha 128 " + `
                        "--lora_dropout 0.05"
} else {
    # Standard training command without DeepSpeed
    $command = "python finetune/trainer.py " + `
                    "--model_name $modelName " + `
                    "--dataset_dir finetune/dataset " + `
                    "--output_dir $outputDir " + `
                    "--num_epochs $trainEpochs " + `
                    "--batch_size $batchSize " + `
                    "--gradient_accumulation_steps $gradAccumSteps " + `
                    "--learning_rate 2e-4 " + `
                    "--logging_steps 10 " + `
                    "--save_steps 100 " + `
                    "--warmup_steps 50 " + `
                    "--max_seq_length 2048 " + `
                    "--lora_r 64 " + `
                    "--lora_alpha 128 " + `
                    "--lora_dropout 0.05"
}

# Run the fine-tuning command
Write-Host "Running: $command"
$result = $null
try {
    $result = Invoke-Expression $command
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "Error running command: $_" -ForegroundColor Red
    $exitCode = 1
}

# Check if the command was successful
if ($exitCode -eq 0) {
    Write-Host "Fine-tuning completed successfully!" -ForegroundColor Green
    Write-Host "Model checkpoint saved to: $outputDir"
} else {
    Write-Host "Fine-tuning failed with exit code $exitCode" -ForegroundColor Red
    if ($result) {
        Write-Host "Command output: $result"
    }
} 