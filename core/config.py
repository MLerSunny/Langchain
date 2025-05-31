import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import jsonschema
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration schema
DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "app": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "environment": {"type": "string", "enum": ["development", "staging", "production"]},
                "debug": {"type": "boolean"},
                "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]}
            },
            "required": ["name", "version", "environment"]
        },
        "rag": {
            "type": "object",
            "properties": {
                "chunk_size": {"type": "integer", "minimum": 100, "maximum": 2000},
                "chunk_overlap": {"type": "integer", "minimum": 0, "maximum": 500},
                "max_chunks": {"type": "integer", "minimum": 1},
                "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "cache_results": {"type": "boolean"},
                "cache_ttl": {"type": "integer", "minimum": 0}
            },
            "required": ["chunk_size", "chunk_overlap", "max_chunks"]
        },
        "llm": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 1},
                "max_tokens": {"type": "integer", "minimum": 1},
                "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                "frequency_penalty": {"type": "number", "minimum": -2, "maximum": 2},
                "presence_penalty": {"type": "number", "minimum": -2, "maximum": 2}
            },
            "required": ["model", "temperature", "max_tokens"]
        },
        "vector_store": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["chroma", "faiss", "pinecone"]},
                "persist_directory": {"type": "string"},
                "collection_name": {"type": "string"},
                "dimension": {"type": "integer", "minimum": 1},
                "index_params": {"type": "object"}
            },
            "required": ["type", "persist_directory"]
        },
        "embeddings": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "cache_dir": {"type": "string"},
                "batch_size": {"type": "integer", "minimum": 1}
            },
            "required": ["model"]
        },
        "security": {
            "type": "object",
            "properties": {
                "api_key": {"type": "string"},
                "allowed_origins": {"type": "array", "items": {"type": "string"}},
                "rate_limit": {"type": "integer", "minimum": 1},
                "max_file_size": {"type": "integer", "minimum": 1}
            },
            "required": ["api_key"]
        }
    },
    "required": ["app", "rag", "llm", "vector_store", "embeddings", "security"]
}

class ConfigError(Exception):
    """Base class for configuration-related errors."""
    pass

class ConfigValidationError(ConfigError):
    """Error raised when configuration validation fails."""
    pass

class ConfigFileError(ConfigError):
    """Error raised when there are issues with the configuration file."""
    pass

@dataclass
class Config:
    """Configuration management class. YAML is now the default expected config format for the project."""
    
    config_path: str
    schema: Dict[str, Any] = field(default_factory=lambda: DEFAULT_SCHEMA)
    config: Dict[str, Any] = field(default_factory=dict)
    last_modified: Optional[datetime] = None
    
    def __post_init__(self):
        """Load and validate configuration after initialization."""
        if not os.path.exists(self.config_path):
            raise ConfigFileError(f"Configuration file not found: {self.config_path}")
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    raise ConfigFileError(f"Unsupported configuration file format: {self.config_path}")
            
            if not isinstance(self.config, dict):
                raise ConfigValidationError("Configuration must be a dictionary")
            
            self.last_modified = datetime.fromtimestamp(os.path.getmtime(self.config_path))
            self.validate_config()
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigFileError(f"Error parsing configuration file: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise ConfigError(f"Failed to load configuration: {str(e)}")
    
    def validate_config(self) -> None:
        """Validate configuration against schema."""
        try:
            jsonschema.validate(instance=self.config, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            error_msg = f"Configuration validation error: {str(e)}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during configuration validation: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if not isinstance(value, dict):
                    return default
                value = value.get(k)
                if value is None:
                    return default
            
            return value
        except Exception as e:
            logger.error(f"Error getting configuration value for key '{key}': {str(e)}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        try:
            keys = key.split('.')
            config = self.config
            
            # Create nested dictionaries if they don't exist
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                elif not isinstance(config[k], dict):
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Validate the updated configuration
            self.validate_config()
            
        except Exception as e:
            error_msg = f"Error setting configuration value for key '{key}': {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            # Create backup of existing file
            if os.path.exists(self.config_path):
                backup_path = f"{self.config_path}.bak"
                shutil.copy2(self.config_path, backup_path)
            
            # Write new configuration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif self.config_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
            
            self.last_modified = datetime.fromtimestamp(os.path.getmtime(self.config_path))
            
        except Exception as e:
            error_msg = f"Error saving configuration: {str(e)}"
            logger.error(error_msg)
            # Restore backup if it exists
            if os.path.exists(f"{self.config_path}.bak"):
                shutil.copy2(f"{self.config_path}.bak", self.config_path)
            raise ConfigError(error_msg)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        try:
            self.load_config()
        except Exception as e:
            error_msg = f"Error reloading configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self.config.copy()  # Return a copy to prevent modification
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        try:
            for key, value in updates.items():
                self.set(key, value)
            self.save()
        except Exception as e:
            error_msg = f"Error updating configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    def reset(self) -> None:
        """Reset configuration to default values."""
        try:
            self.config = {}
            self.save()
        except Exception as e:
            error_msg = f"Error resetting configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigError(error_msg)
    
    def validate_value(self, key: str, value: Any) -> bool:
        """Validate a single configuration value."""
        try:
            # Get the schema for the specific key
            keys = key.split('.')
            schema = self.schema
            
            for k in keys:
                if 'properties' in schema:
                    schema = schema['properties'].get(k, {})
                else:
                    return False
            
            # Validate the value against the schema
            jsonschema.validate(instance=value, schema=schema)
            return True
            
        except jsonschema.exceptions.ValidationError:
            return False
        except Exception as e:
            logger.error(f"Error validating configuration value: {str(e)}")
            return False
    
    def get_environment(self) -> str:
        """Get current environment."""
        return self.get('app.environment', 'development')
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == 'development'
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.get_environment() == 'staging'
    
    def get_log_level(self) -> str:
        """Get configured log level."""
        return self.get('app.log_level', 'INFO')
    
    def get_api_key(self) -> str:
        """Get API key."""
        return self.get('security.api_key')
    
    def get_allowed_origins(self) -> List[str]:
        """Get allowed origins for CORS."""
        return self.get('security.allowed_origins', [])
    
    def get_rate_limit(self) -> int:
        """Get rate limit configuration."""
        return self.get('security.rate_limit', 100)
    
    def get_max_file_size(self) -> int:
        """Get maximum file size configuration."""
        return self.get('security.max_file_size', 10 * 1024 * 1024)  # 10MB default
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self.get('rag', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.get('llm', {})
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        return self.get('vector_store', {})
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self.get('embeddings', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.get('security', {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.get('app', {})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            "enabled": self.get('rag.cache_results', False),
            "ttl": self.get('rag.cache_ttl', 3600),
            "directory": self.get('embeddings.cache_dir', 'data/cache')
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "name": self.get('llm.model'),
            "temperature": self.get('llm.temperature', 0.7),
            "max_tokens": self.get('llm.max_tokens', 1000),
            "top_p": self.get('llm.top_p', 1.0),
            "frequency_penalty": self.get('llm.frequency_penalty', 0.0),
            "presence_penalty": self.get('llm.presence_penalty', 0.0)
        }
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get document chunking configuration."""
        return {
            "chunk_size": self.get('rag.chunk_size', 1000),
            "chunk_overlap": self.get('rag.chunk_overlap', 200),
            "max_chunks": self.get('rag.max_chunks', 10)
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return {
            "similarity_threshold": self.get('rag.similarity_threshold', 0.7),
            "max_results": self.get('rag.max_chunks', 10)
        }
    
    def get_index_config(self) -> Dict[str, Any]:
        """Get vector index configuration."""
        return {
            "type": self.get('vector_store.type'),
            "dimension": self.get('vector_store.dimension', 1536),
            "params": self.get('vector_store.index_params', {})
        }
    
    def get_persist_config(self) -> Dict[str, Any]:
        """Get persistence configuration."""
        return {
            "directory": self.get('vector_store.persist_directory', 'data/vector_store'),
            "collection": self.get('vector_store.collection_name', 'default')
        }
    
    def get_batch_config(self) -> Dict[str, Any]:
        """Get batch processing configuration."""
        return {
            "size": self.get('embeddings.batch_size', 32),
            "timeout": self.get('llm.timeout', 30)
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "enabled": self.get('app.monitoring.enabled', False),
            "interval": self.get('app.monitoring.interval', 60),
            "metrics": self.get('app.monitoring.metrics', []),
            "alerts": self.get('app.monitoring.alerts', {})
        }
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return {
            "retry_attempts": self.get('app.error_handling.retry_attempts', 3),
            "retry_delay": self.get('app.error_handling.retry_delay', 1),
            "fallback_responses": self.get('app.error_handling.fallback_responses', {})
        }
    
    def get_prompt_config(self) -> Dict[str, Any]:
        """Get prompt configuration."""
        return {
            "system": self.get('app.prompts.system', ''),
            "user": self.get('app.prompts.user', ''),
            "assistant": self.get('app.prompts.assistant', '')
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flag configuration."""
        return self.get('app.feature_flags', {})
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        return self.get(f'app.feature_flags.{feature}', False)
    
    def get_version(self) -> str:
        """Get application version."""
        return self.get('app.version', '0.1.0')
    
    def get_name(self) -> str:
        """Get application name."""
        return self.get('app.name', 'Insurance Assistant')
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('app.debug', False)
    
    def get_last_modified(self) -> Optional[datetime]:
        """Get last modification time of configuration file."""
        return self.last_modified
    
    def has_changed(self) -> bool:
        """Check if configuration file has changed since last load."""
        if not self.last_modified:
            return True
        
        try:
            current_mtime = datetime.fromtimestamp(os.path.getmtime(self.config_path))
            return current_mtime > self.last_modified
        except OSError:
            return True 