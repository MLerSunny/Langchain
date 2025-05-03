"""
Gunicorn configuration file for production deployment.
"""

import multiprocessing

# Gunicorn config variables
bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
accesslog = "-"
errorlog = "-"
loglevel = "info"

# For development
reload = False

# For debugging
capture_output = True 