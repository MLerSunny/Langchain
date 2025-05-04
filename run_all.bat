@echo off
setlocal enabledelayedexpansion

echo ================================================
echo LANGCHAIN ALL-IN-ONE LAUNCHER 
echo ================================================
echo This script will:
echo 1. Kill any existing processes
echo 2. Set up the vector database
echo 3. Start both the RAG server and Streamlit app
echo ================================================

:: Define paths
set "BASE_DIR=%~dp0"
set "PYTHON_PATH=%BASE_DIR%.venv\Scripts\python.exe"
set "SERVE_PATH=%BASE_DIR%scripts\serve.py"
set "STREAMLIT_PATH=%BASE_DIR%streamlit_app.py"
set "DATA_DIR=%BASE_DIR%data"
set "CHROMA_DIR=%BASE_DIR%data\chroma"

echo Step 1: Stopping existing Python processes...
taskkill /F /IM python.exe /T 2>nul
echo.

echo Step 2: Creating necessary directories...
if not exist "%DATA_DIR%" (
    mkdir "%DATA_DIR%"
)
if not exist "%CHROMA_DIR%" (
    mkdir "%CHROMA_DIR%"
)
echo.

echo Step 3: Creating Streamlit and environment configs...
:: Create Streamlit config with IP address instead of hostname
if not exist "%BASE_DIR%.streamlit" mkdir "%BASE_DIR%.streamlit"
(
echo [server]
echo headless = false
echo enableCORS = true
echo enableXsrfProtection = true
echo port = 8501
echo [browser]
echo serverAddress = "127.0.0.1"
echo serverPort = 8501
) > "%BASE_DIR%.streamlit\config.toml"

:: Create environment config
(
echo ollama_base_url=http://127.0.0.1:11434
echo VECTOR_DB=chroma
echo CHROMA_PERSIST_DIRECTORY=
) > "%BASE_DIR%.env"
echo.

echo Step 4: Setting up vector database...
:: Run the pre-created Python script to initialize the vector DB
echo Running vector DB initialization...
"%PYTHON_PATH%" "%BASE_DIR%init_vector_db.py"
if %errorlevel% neq 0 (
    echo ERROR: Failed to initialize vector database!
    echo Please check the Python error output above.
    pause
    exit /b 1
)
echo.

echo Step 5: Starting RAG API server...
start "RAG API Server" cmd /c "cd /d %BASE_DIR% && %PYTHON_PATH% %SERVE_PATH%"
echo Waiting for RAG server to initialize...
timeout /t 10 /nobreak > nul

echo Step 6: Starting Streamlit app...
start "Streamlit App" cmd /c "cd /d %BASE_DIR% && %PYTHON_PATH% -m streamlit run %STREAMLIT_PATH%"
echo Waiting for Streamlit to initialize...
timeout /t 8 /nobreak > nul

echo Step 7: Opening browser...
:: Commenting out to prevent duplicate browser windows - Streamlit already opens one
:: start http://127.0.0.1:8501

echo.
echo ================================================
echo ALL SERVICES STARTED SUCCESSFULLY
echo ================================================
echo.
echo Your Langchain app is now running:
echo - Streamlit UI: http://127.0.0.1:8501
echo - RAG API: http://127.0.0.1:8000
echo.
echo To stop all services, just close all terminal windows
echo or run: taskkill /F /IM python.exe
echo.
echo Press any key to exit this script...
pause > nul 