# Setup script for RAG + Fine-tuning environment on Windows
# Installs CUDA 12, PyTorch + cu121, oneAPI, and bits-and-bytes

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit
}

# Create a temporary directory for downloads
$tempDir = "C:\temp"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir | Out-Null
}

# Set execution policy to allow running scripts
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

Write-Host "Setting up RAG + Fine-tuning environment..." -ForegroundColor Cyan

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $command) { return $true }
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

# Install Chocolatey if not already installed
if (-not (Test-CommandExists choco)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    # Reload PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# Install CUDA 12.1
if (-not (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1")) {
    Write-Host "Installing CUDA 12.1..." -ForegroundColor Yellow
    # Download CUDA installer
    $cudaInstallerUrl = "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_windows.exe"
    $cudaInstallerPath = "$tempDir\cuda_installer.exe"
    
    if (-not (Test-Path $cudaInstallerPath)) {
        Write-Host "Downloading CUDA installer (this may take a while)..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $cudaInstallerUrl -OutFile $cudaInstallerPath
    }
    
    # Install CUDA silently
    Write-Host "Installing CUDA 12.1 (this may take a while)..." -ForegroundColor Yellow
    Start-Process -Wait -FilePath $cudaInstallerPath -ArgumentList "-s"
    
    # Add CUDA to PATH
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    $cudaLibPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp"
    
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (-not $currentPath.Contains($cudaPath)) {
        [Environment]::SetEnvironmentVariable("Path", $currentPath + ";" + $cudaPath + ";" + $cudaLibPath, "Machine")
    }
    
    Write-Host "CUDA 12.1 installed successfully" -ForegroundColor Green
} else {
    Write-Host "CUDA 12.1 is already installed" -ForegroundColor Green
}

# Install Python 3.11 if not already installed
if (-not (Test-CommandExists python) -or (python --version) -notlike "*3.11*") {
    Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
    choco install python311 -y
    # Reload PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# Install PyTorch with CUDA 12.1
Write-Host "Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install oneAPI MKL
Write-Host "Installing Intel oneAPI MKL..." -ForegroundColor Yellow
pip install --upgrade intel-extension-for-pytorch

# Install bitsandbytes for quantization
Write-Host "Installing bitsandbytes..." -ForegroundColor Yellow
pip install bitsandbytes

# Install other dependencies
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install -U peft transformers accelerate scipy datasets einops sentencepiece protobuf xformers

# Install vLLM
Write-Host "Installing vLLM..." -ForegroundColor Yellow
pip install vllm

# Install FastAPI and related packages
Write-Host "Installing FastAPI and related packages..." -ForegroundColor Yellow
pip install fastapi uvicorn aiohttp

# Install LangChain
Write-Host "Installing LangChain and Chroma..." -ForegroundColor Yellow
pip install langchain langchain_community langchain_experimental langchain-chroma

# Install Ollama
if (-not (Test-CommandExists ollama)) {
    Write-Host "Installing Ollama..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://github.com/ollama/ollama/releases/download/v0.1.24/ollama-windows-amd64.zip" -OutFile "$tempDir\ollama.zip"
    Expand-Archive -Path "$tempDir\ollama.zip" -DestinationPath "$env:ProgramFiles\Ollama" -Force
    # Add Ollama to PATH
    $ollamaPath = "$env:ProgramFiles\Ollama"
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (-not $currentPath.Contains($ollamaPath)) {
        [Environment]::SetEnvironmentVariable("Path", $currentPath + ";" + $ollamaPath, "Machine")
    }
}

# Clean up temp files
if (Test-Path "$tempDir\cuda_installer.exe") {
    Remove-Item "$tempDir\cuda_installer.exe" -Force
}
if (Test-Path "$tempDir\ollama.zip") {
    Remove-Item "$tempDir\ollama.zip" -Force
}

Write-Host "Environment setup completed successfully!" -ForegroundColor Green
Write-Host "Please restart your computer to ensure all changes take effect." -ForegroundColor Yellow 