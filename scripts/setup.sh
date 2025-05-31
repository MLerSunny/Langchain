#!/bin/bash
# Setup script for RAG + Fine-tuning System

# Print colorful messages
print_green() {
    echo -e "\033[32m$1\033[0m"
}

print_yellow() {
    echo -e "\033[33m$1\033[0m"
}

print_red() {
    echo -e "\033[31m$1\033[0m"
}

# Check if Python is installed
if ! command -v python &> /dev/null; then
    print_red "Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    print_red "Python version $python_version is not supported. Please install Python 3.8 or higher."
    exit 1
fi

print_green "Setting up environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_yellow "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install requirements
print_yellow "Installing requirements..."
pip install -r requirements.txt

# Run environment setup script
print_yellow "Setting up environment variables..."
python scripts/setup_env.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    print_green "Setup completed successfully!"
    print_green "You can now start the servers:"
    print_yellow "1. RAG Server: python scripts/start_rag_server.py"
    print_yellow "2. Fine-tuning Server: python scripts/start_fine_tuning_server.py"
    print_yellow "3. Streamlit App: streamlit run streamlit_app.py"
else
    print_red "Setup failed. Please check the error messages above."
    exit 1
fi 