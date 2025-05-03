# Windows Installation Guide

This guide provides detailed instructions for setting up the DeepSeek RAG + Fine-tuning System on Windows.

## System Requirements

- Windows 10 or 11
- Python 3.9+ (3.10 recommended)
- NVIDIA GPU with at least 8GB VRAM (recommended)
- At least 16GB of system RAM
- 10GB+ free disk space

## Step 1: Install Python

1. Download Python 3.10 from [python.org](https://www.python.org/downloads/windows/)
2. During installation:
   - Check "Add Python to PATH"
   - Choose "Customize installation"
   - Ensure pip is selected
   - Install for all users

## Step 2: Install Git

1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Install with default options

## Step 3: Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

## Step 4: Install Visual C++ Build Tools

OCR and PDF processing libraries require Visual C++ Build Tools:

1. Download the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer and select:
   - C++ build tools
   - Windows 10 SDK

## Step 5: Install Tesseract OCR (for OCR support)

1. Download the [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. During installation:
   - Install to `C:\Program Files\Tesseract-OCR`
   - Check "Add to PATH"
   - Select additional languages if needed

4. **Verify installation**:
   ```bash
   tesseract --version
   ```

## Step 6: Install Poppler (for PDF processing)

1. Download [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
2. Extract to `C:\Program Files\poppler`
3. Add to PATH:
   - Right-click on "This PC" > Properties > Advanced system settings > Environment Variables
   - Edit the "Path" variable for either your user or system
   - Add `C:\Program Files\poppler\bin`

4. **Verify installation**:
   ```bash
   pdfinfo -v
   ```

## Step 7: Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Step 8: Install Dependencies

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Step 9: Install Ollama

1. Download [Ollama for Windows](https://ollama.com/download/windows)
2. Run the installer
3. Ollama will be available in your system tray

## Step 10: Pull DeepSeek Model

```bash
ollama pull deepseek-coder:7b-instruct-v1.5
```

## Step 11: Start the Application

### Option 1: Using Makefile with PowerShell

```powershell
make streamlit
```

### Option 2: Direct Command

```powershell
streamlit run streamlit_app.py
```

## Troubleshooting

### OCR Installation Issues

If you encounter issues with OCR:

1. Ensure Tesseract is properly installed
2. Verify the PATH environment variable includes the Tesseract directory
3. Try setting the environment variable:
   ```powershell
   $env:TESSDATA_PREFIX="C:\Program Files\Tesseract-OCR\tessdata"
   ```

### PDF Processing Issues

If you encounter issues with PDF processing:

1. Ensure Poppler is properly installed
2. Verify the PATH environment variable includes the Poppler bin directory
3. Try reinstalling with:
   ```bash
   pip uninstall pdf2image
   pip install pdf2image
   ```

### Memory Issues

If you encounter memory errors:

1. Reduce batch size and max_chunks in the Streamlit interface
2. Ensure you have enough free RAM (at least 8GB)
3. Try smaller document sets
4. For large PDF files, split them into smaller files before uploading

### CUDA Issues

If you encounter CUDA errors:

1. Ensure you have a compatible NVIDIA GPU
2. Install the latest NVIDIA drivers
3. Install CUDA Toolkit 11.8
4. Verify torch is using CUDA:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Additional Resources

- [DeepSeek Documentation](https://github.com/deepseek-ai)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Ollama Documentation](https://ollama.com/docs)

For any other issues, please check the project's GitHub issues or open a new issue. 