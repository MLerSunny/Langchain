"""
Logging utility for the application.

This module provides a consistent logging setup for the entire application.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Name of the logger.
        level: Logging level. Default is INFO.
        log_file: Optional path to log file.
        console_output: Whether to output to console. Default is True.
        format_string: Optional format string for log messages.

    Returns:
        Configured logger.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    formatter = logging.Formatter(format_string)
    
    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if console_output is True
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Default application logger
app_logger = setup_logger("app") 