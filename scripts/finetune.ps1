# PowerShell script for fine-tuning DeepSeek-LLM
# Usage: .\scripts\finetune.ps1

param (
    [string]$model_name = "deepseek-llm:7b",
    [string]$dataset_path = "finetune/dataset/insurance_conversations.json",
    [string]$output_dir = "data/models",
    [string]$learning_rate = "2e-5",
    [int]$batch_size = 1,
    [int]$gradient_accumulation_steps = 4,
    [int]$num_epochs = 3,
    [int]$max_steps = -1,
    [int]$logging_steps = 10,
    [int]$save_steps = 100,
    [int]$warmup_steps = 50,
    [int]$lora_r = 64,
    [int]$lora_alpha = 128,
    [float]$lora_dropout = 0.05,
    [bool]$use_lora = $true,
    [bool]$use_4bit = $true,
    [int]$max_seq_length = 2048
)

Write-Host "Starting DeepSeek-LLM fine-tuning..." -ForegroundColor Cyan
Write-Host "Model: $model_name" -ForegroundColor Cyan
Write-Host "Dataset: $dataset_path" -ForegroundColor Cyan
Write-Host "Output directory: $output_dir" -ForegroundColor Cyan

# Ensure the dataset file exists
if (-not (Test-Path $dataset_path)) {
    Write-Error "Dataset file $dataset_path does not exist!"
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $output_dir)) {
    New-Item -Path $output_dir -ItemType Directory -Force | Out-Null
    Write-Host "Created output directory: $output_dir" -ForegroundColor Green
}

# Set environment variables for training
$env:TOKENIZERS_PARALLELISM = "false"

# Check for GPU availability
$gpuAvailable = $false
try {
    $gpuInfo = nvidia-smi
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NVIDIA GPU detected, using CUDA" -ForegroundColor Green
        $env:CUDA_VISIBLE_DEVICES = "0"
        $gpuAvailable = $true
    }
} catch {
    Write-Host "No NVIDIA GPU detected, using CPU only (this will be slow)" -ForegroundColor Yellow
    $use_4bit = $false
}

# Build the Python command
$command = "python scripts/finetune.py " + `
    "--model_name_or_path `"$model_name`" " + `
    "--dataset_path `"$dataset_path`" " + `
    "--output_dir `"$output_dir`" " + `
    "--learning_rate $learning_rate " + `
    "--per_device_train_batch_size $batch_size " + `
    "--gradient_accumulation_steps $gradient_accumulation_steps " + `
    "--num_train_epochs $num_epochs " + `
    "--max_steps $max_steps " + `
    "--logging_steps $logging_steps " + `
    "--save_steps $save_steps " + `
    "--warmup_steps $warmup_steps " + `
    "--use_lora `$$use_lora " + `
    "--lora_rank $lora_r " + `
    "--lora_alpha $lora_alpha " + `
    "--lora_dropout $lora_dropout " + `
    "--use_4bit `$$use_4bit " + `
    "--max_seq_length $max_seq_length " + `
    "--report_to tensorboard"

# Run the fine-tuning command
Write-Host "Running: $command" -ForegroundColor Cyan
try {
    Invoke-Expression $command
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "Error running command: $_" -ForegroundColor Red
    $exitCode = 1
}

# Check if the command was successful
if ($exitCode -eq 0) {
    Write-Host "Fine-tuning completed successfully!" -ForegroundColor Green
    Write-Host "Model checkpoint saved to: $output_dir" -ForegroundColor Green
} else {
    Write-Host "Fine-tuning failed with exit code $exitCode" -ForegroundColor Red
    exit 1
} 