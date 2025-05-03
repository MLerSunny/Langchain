FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"] 