"""
Gunicorn configuration file for production deployment.
"""

import multiprocessing
import os
from core.settings import settings

# Gunicorn config variables
bind = f"{settings.host}:{settings.fastapi_port}"
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('GUNICORN_WORKER_CLASS', "uvicorn.workers.UvicornWorker")
timeout = int(os.getenv('GUNICORN_TIMEOUT', 120))
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', 5))
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', 50))
accesslog = os.getenv('GUNICORN_ACCESS_LOG', "-")
errorlog = os.getenv('GUNICORN_ERROR_LOG', "-")
loglevel = os.getenv('GUNICORN_LOG_LEVEL', "info")

# For development
reload = os.getenv('GUNICORN_RELOAD', 'false').lower() == 'true'

# For debugging
capture_output = os.getenv('GUNICORN_CAPTURE_OUTPUT', 'true').lower() == 'true' 