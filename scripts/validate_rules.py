#!/usr/bin/env python3
"""
Validation script to ensure codebase follows the rules defined in RULES.md.
"""

import os
import yaml
import logging
import ast
from pathlib import Path
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleValidator:
    def __init__(self, config_path: str = "config/rag.yaml"):
        self.config_path = config_path
        self.root_dir = Path(__file__).parent.parent
        self.config = self._load_config()
        self.hardcoded_values: Set[str] = set()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return {}

    def validate_llm_backend(self) -> bool:
        """Validate LLM backend configuration."""
        try:
            llm_config = self.config.get('llm', {})
            required_fields = [
                'backend', 'model', 'temperature', 'max_tokens',
                'top_p', 'frequency_penalty', 'presence_penalty',
                'repetition_penalty'
            ]
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in llm_config]
            if missing_fields:
                logger.error(f"Missing required LLM fields: {missing_fields}")
                return False
                
            # Check backend value
            if llm_config['backend'] not in ['huggingface', 'vllm']:
                logger.error(f"Invalid LLM backend: {llm_config['backend']}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating LLM backend: {e}")
            return False

    def validate_sharegpt_config(self) -> bool:
        """Validate ShareGPT configuration."""
        try:
            sharegpt_config = self.config.get('sharegpt', {})
            query_config = sharegpt_config.get('query', {})
            
            required_fields = [
                'query_template', 'response_format', 'questions_per_chunk',
                'min_question_length', 'max_question_length'
            ]
            
            missing_fields = [field for field in required_fields if field not in query_config]
            if missing_fields:
                logger.error(f"Missing required ShareGPT fields: {missing_fields}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating ShareGPT config: {e}")
            return False

    def validate_directory_structure(self) -> bool:
        """Validate required directory structure."""
        try:
            required_dirs = [
                'core/llm_backends',
                'scripts',
                'data/training/data',
                'data/models',
                'data/logs',
                'data/cache',
                'data/metrics',
                'data/jobs'
            ]
            
            for dir_path in required_dirs:
                full_path = self.root_dir / dir_path
                if not full_path.exists():
                    logger.error(f"Missing required directory: {dir_path}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error validating directory structure: {e}")
            return False

    def validate_file_structure(self) -> bool:
        """Validate required file structure."""
        try:
            required_files = [
                'core/llm_backends/base.py',
                'core/llm_backends/huggingface_backend.py',
                'core/llm_backends/vllm_backend.py',
                'core/llm_backends/factory.py',
                'core/rag_engine.py',
                'core/settings.py',
                'scripts/convert_to_sharegpt.py',
                'scripts/batch_ingest.py',
                'scripts/simple_vector_ingest.py'
            ]
            
            for file_path in required_files:
                full_path = self.root_dir / file_path
                if not full_path.exists():
                    logger.error(f"Missing required file: {file_path}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error validating file structure: {e}")
            return False

    def validate_error_handling(self) -> bool:
        """Validate error handling configuration."""
        try:
            error_config = self.config.get('error_handling', {})
            fallback_config = error_config.get('fallback_responses', {})
            
            required_fields = [
                'retry_attempts', 'retry_delay',
                'fallback_responses.generation_error',
                'fallback_responses.retrieval_error',
                'fallback_responses.validation_error'
            ]
            
            missing_fields = []
            for field in required_fields:
                if '.' in field:
                    parent, child = field.split('.')
                    if parent not in error_config or child not in error_config[parent]:
                        missing_fields.append(field)
                elif field not in error_config:
                    missing_fields.append(field)
                    
            if missing_fields:
                logger.error(f"Missing required error handling fields: {missing_fields}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating error handling: {e}")
            return False

    def validate_monitoring(self) -> bool:
        """Validate monitoring configuration."""
        try:
            monitoring_config = self.config.get('monitoring', {})
            metrics = monitoring_config.get('metrics', [])
            
            required_metrics = ['latency', 'throughput', 'error_rate', 'token_usage']
            missing_metrics = [metric for metric in required_metrics if metric not in metrics]
            
            if missing_metrics:
                logger.error(f"Missing required metrics: {missing_metrics}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating monitoring: {e}")
            return False

    def validate_paths(self) -> bool:
        """Validate paths configuration."""
        try:
            paths_config = self.config.get('paths', {})
            required_paths = [
                'data_dir', 'training_data_dir', 'models_dir',
                'logs_dir', 'cache_dir', 'metrics_dir', 'jobs_dir'
            ]
            
            missing_paths = [path for path in required_paths if path not in paths_config]
            if missing_paths:
                logger.error(f"Missing required paths: {missing_paths}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating paths: {e}")
            return False

    def validate_config_driven_approach(self) -> bool:
        """Validate config-driven approach rules."""
        try:
            # Check config section exists
            config_section = self.config.get('config', {})
            required_config_fields = ['version', 'environment', 'debug', 'strict_mode']
            missing_fields = [field for field in required_config_fields if field not in config_section]
            if missing_fields:
                logger.error(f"Missing required config fields: {missing_fields}")
                return False

            # Check for hardcoded values in Python files
            python_files = self._find_python_files()
            for file_path in python_files:
                if not self._check_file_for_hardcoded_values(file_path):
                    return False

            # Check for Config class usage
            if not self._check_config_class_usage():
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating config-driven approach: {e}")
            return False

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def _check_file_for_hardcoded_values(self, file_path: Path) -> bool:
        """Check a Python file for hardcoded configuration values."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
                # Check for string literals that might be config values
                for node in ast.walk(tree):
                    if isinstance(node, ast.Str):
                        value = node.s
                        # Skip docstrings and short strings
                        if len(value) > 10 and not value.startswith('"""') and not value.startswith("'''"):
                            self.hardcoded_values.add(f"{file_path}:{node.lineno}: {value}")
                
                # Check for dictionary literals that might be config
                for node in ast.walk(tree):
                    if isinstance(node, ast.Dict):
                        for key in node.keys:
                            if isinstance(key, ast.Str):
                                self.hardcoded_values.add(f"{file_path}:{key.lineno}: {key.s}")
                
                return True
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False

    def _check_config_class_usage(self) -> bool:
        """Check if Config class is properly used."""
        try:
            config_imports = 0
            config_usage = 0
            
            for file_path in self._find_python_files():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'from core.config import Config' in content or 'import core.config' in content:
                        config_imports += 1
                    if 'Config(' in content:
                        config_usage += 1
            
            if config_imports == 0 or config_usage == 0:
                logger.error("Config class is not being used properly")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking Config class usage: {e}")
            return False

    def run_all_validations(self) -> bool:
        """Run all validations and return overall status."""
        validations = [
            self.validate_llm_backend,
            self.validate_sharegpt_config,
            self.validate_directory_structure,
            self.validate_file_structure,
            self.validate_error_handling,
            self.validate_monitoring,
            self.validate_paths,
            self.validate_config_driven_approach
        ]
        
        results = []
        for validation in validations:
            try:
                result = validation()
                results.append(result)
                if not result:
                    logger.error(f"Validation failed: {validation.__name__}")
            except Exception as e:
                logger.error(f"Error running validation {validation.__name__}: {e}")
                results.append(False)
        
        # Report hardcoded values if found
        if self.hardcoded_values:
            logger.warning("Found potential hardcoded configuration values:")
            for value in sorted(self.hardcoded_values):
                logger.warning(f"  {value}")
                
        return all(results)

def main():
    """Main entry point for the validation script."""
    validator = RuleValidator()
    if validator.run_all_validations():
        logger.info("All validations passed successfully!")
        return 0
    else:
        logger.error("Some validations failed. Please check the logs for details.")
        return 1

if __name__ == "__main__":
    exit(main()) 