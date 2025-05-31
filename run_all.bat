@echo off
echo Setting up Python environment...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Create necessary directories
echo Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\chroma" mkdir data\chroma
if not exist "data\training" mkdir data\training
if not exist "data\eval" mkdir data\eval
if not exist "data\models" mkdir data\models
if not exist "logs" mkdir logs

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

:: Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%
set DATA_PATH=data\raw
set CHROMA_PATH=.chroma
set FASTAPI_PORT=8000
set HOST=localhost
set MODEL_NAME=deepseek-llm:7b
set CONTEXT_WINDOW=8192
set MAX_TOKENS=2048
set TEMPERATURE=0.1
set CHROMA_PERSIST_DIRECTORY=data\chroma
set EMBEDDING_DIMENSION=768
set RAG_CONFIG_PATH=config\rag.yaml
set TRAINING_DATASET_PATH=data\training
set EVAL_DATASET_PATH=data\eval
set OUTPUT_DIR=data\models
set LEARNING_RATE=0.00002
set BATCH_SIZE=1
set NUM_EPOCHS=3

:: Check if required files exist
if not exist "streamlit_app.py" (
    echo Error: streamlit_app.py not found!
    pause
    exit /b 1
)

if not exist "scripts\start_fine_tuning_server.py" (
    echo Error: scripts\start_fine_tuning_server.py not found!
    pause
    exit /b 1
)

:: Launch Streamlit UI in a new window
echo Starting Streamlit UI...
start "Streamlit UI" cmd /k "call venv\Scripts\activate.bat && streamlit run streamlit_app.py"

:: Launch RAG server in a new window
echo Starting RAG server...
start "RAG Server" cmd /k "call venv\Scripts\activate.bat && python scripts\start_rag_server.py"

:: Launch FastAPI backend server in a new window
start "FastAPI Backend" cmd /k "call venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"

:: Launch LLM fine-tuning server in a new window
echo Starting LLM fine-tuning server...
start "Fine-tuning Server" cmd /k "call venv\Scripts\activate.bat && python scripts\start_fine_tuning_server.py"

echo.
echo All services have been started in separate windows:
echo - Streamlit UI: http://localhost:8501
echo - RAG Server: http://localhost:8000
echo - Fine-tuning Server: http://localhost:8001
echo.
echo Press any key to close this window...
pause

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat 