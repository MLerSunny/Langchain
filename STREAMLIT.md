# DeepSeek RAG + Fine-tuning System UI

This Streamlit application provides a user-friendly interface for the DeepSeek RAG + Fine-tuning System. It allows you to:

1. **Convert Documents to ShareGPT Format**: Upload various document formats and convert them to ShareGPT-style conversation data for fine-tuning.
2. **Query the RAG System**: Interact with the RAG system to ask questions about your documents.
3. **Fine-tune DeepSeek Models**: Configure and run fine-tuning of DeepSeek models through an intuitive interface.
4. **Configure System Settings**: Adjust various system settings and configurations.

## Running the Application

You can run the Streamlit application using:

```bash
# Using Makefile
make streamlit

# Or directly with Streamlit
streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501 by default.

## Converting Documents to ShareGPT Format

The document conversion process follows these steps:

1. **Upload Documents**: Upload files in various formats (PDF, DOCX, CSV, TXT, HTML, etc.).
2. **Configure Processing**: Set chunk size and overlap for document splitting.
3. **Configure Conversion**: Customize the system prompt and question generation parameters.
4. **Process and Review**: View the processing results, including document statistics and preview of the generated ShareGPT format data.
5. **Download or Save**: Download the generated data or save it to the training directory.

### Supported File Formats

- PDF (.pdf)
- Word Documents (.docx, .doc)
- CSV (.csv)
- Text (.txt)
- HTML (.html, .htm)
- JSON (.json)
- And more through the UnstructuredFileLoader

## System Requirements

- Python 3.9+
- Dependencies listed in requirements.txt
- Ollama for LLM inference (for question generation)

## Customizing the UI

You can customize the Streamlit UI by modifying the `.streamlit/config.toml` file.

## Adding New Features

The application is designed with modularity in mind. To add new features:

1. Create new tab functions in `streamlit_app.py`
2. Add supporting functions as needed
3. Register the new tab in the `main()` function

## Troubleshooting

**Question generation fails**: Ensure Ollama is running and the DeepSeek model is available.

**File upload issues**: Check that the file format is supported and the file size is within the limits set in `.streamlit/config.toml`.

**Processing hangs**: For large documents, the processing may take some time. Check the logs for progress updates. 