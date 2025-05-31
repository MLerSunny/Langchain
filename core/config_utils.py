"""
Common configuration utilities for loading and validating configuration files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Type, TypeVar, Optional, Union
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def load_config(config_path: str, schema_class: Type[T]) -> Optional[T]:
    """
    Load and validate a configuration file against a Pydantic schema.
    
    Args:
        config_path: Path to the configuration file
        schema_class: Pydantic model class for validation
        
    Returns:
        Validated configuration object or None if loading fails
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            logger.error(f"Empty configuration file: {config_path}")
            return None
            
        # Validate against schema
        try:
            config = schema_class(**config_data)
            return config
        except ValidationError as e:
            logger.error(f"Configuration validation error in {config_path}: {str(e)}")
            return None
            
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        return None

def create_config_directory(config_path: str) -> bool:
    """
    Create directory for configuration file if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        directory = os.path.dirname(config_path)
        if not directory:
            return True
            
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating configuration directory: {str(e)}")
        return False

def save_config(config_path: str, config_data: Dict[str, Any]) -> bool:
    """
    Save configuration data to a YAML file.
    
    Args:
        config_path: Path to save the configuration file
        config_data: Configuration data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not create_config_directory(config_path):
            return False
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        return False

def get_config_path(base_path: str, config_name: str) -> str:
    """
    Get absolute path for a configuration file.
    
    Args:
        base_path: Base directory for configurations
        config_name: Name of the configuration file
        
    Returns:
        Absolute path to the configuration file
    """
    return os.path.abspath(os.path.join(base_path, config_name))

def validate_config_paths(config_paths: Dict[str, str]) -> Dict[str, bool]:
    """
    Validate that all configuration files exist.
    
    Args:
        config_paths: Dictionary of configuration name to path mapping
        
    Returns:
        Dictionary of configuration name to existence status
    """
    return {
        name: os.path.isfile(path)
        for name, path in config_paths.items()
    }

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary with override values
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result 