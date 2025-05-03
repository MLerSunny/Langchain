#!/bin/bash
# Setup script for RAG + Fine-tuning environment on Linux
# Installs CUDA 12, PyTorch + cu121, and required dependencies

# Check if running as root/sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "Please run this script as sudo"
    exit 1
fi

# Create a temporary directory for downloads
TEMP_DIR="/tmp/rag-setup"
mkdir -p $TEMP_DIR

echo "Setting up RAG + Fine-tuning environment..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Update package lists
apt-get update

# Install required dependencies
apt-get install -y build-essential wget curl git

# Install CUDA 12.1 if not already installed
if [ ! -d "/usr/local/cuda-12.1" ]; then
    echo "Installing CUDA 12.1..."
    
    # Download CUDA installer
    CUDA_INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run"
    CUDA_INSTALLER_PATH="$TEMP_DIR/cuda_installer.run"
    
    if [ ! -f "$CUDA_INSTALLER_PATH" ]; then
        echo "Downloading CUDA installer (this may take a while)..."
        wget -O $CUDA_INSTALLER_PATH $CUDA_INSTALLER_URL
    fi
    
    # Install CUDA silently
    echo "Installing CUDA 12.1 (this may take a while)..."
    chmod +x $CUDA_INSTALLER_PATH
    $CUDA_INSTALLER_PATH --silent --toolkit
    
    # Add CUDA to PATH
    echo 'export PATH="/usr/local/cuda-12.1/bin:$PATH"' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"' >> /etc/profile.d/cuda.sh
    source /etc/profile.d/cuda.sh
    
    echo "CUDA 12.1 installed successfully"
else
    echo "CUDA 12.1 is already installed"
fi

# Install Python 3.11 if not already installed
if ! command_exists python3.11; then
    echo "Installing Python 3.11..."
    apt-get install -y python3.11 python3.11-dev python3.11-venv python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    update-alternatives --set python3 /usr/bin/python3.11
fi

# Create and activate virtual environment
VENV_DIR="/opt/rag-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Intel oneAPI MKL
echo "Installing Intel oneAPI MKL..."
pip install --upgrade intel-extension-for-pytorch

# Install bitsandbytes for quantization
echo "Installing bitsandbytes..."
pip install bitsandbytes

# Install other dependencies
echo "Installing other dependencies..."
pip install -U peft transformers accelerate scipy datasets einops sentencepiece protobuf xformers

# Install vLLM
echo "Installing vLLM..."
pip install vllm

# Install FastAPI and related packages
echo "Installing FastAPI and related packages..."
pip install fastapi uvicorn aiohttp

# Install LangChain
echo "Installing LangChain and Chroma..."
pip install langchain langchain_community langchain_experimental langchain-chroma

# Install Ollama
if ! command_exists ollama; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Clean up temp files
rm -rf $TEMP_DIR

echo "Environment setup completed successfully!"
echo "Please restart your terminal or source /etc/profile.d/cuda.sh to ensure all changes take effect." 