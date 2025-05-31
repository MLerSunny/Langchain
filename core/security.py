import re
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import jwt
from functools import wraps
import requests
from ratelimit import limits, sleep_and_retry
import os
import json
import yaml
from pathlib import Path

from core.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Export SecurityManager
__all__ = ['SecurityManager', 'SecurityError', 'InputValidationError', 'RateLimitError', 'TokenError', 'FileValidationError']

# Constants from rag.yaml
MAX_REQUESTS_PER_MINUTE = settings.get('security.rate_limit', 60)
MAX_REQUESTS_PER_HOUR = settings.get('security.rate_limit', 1000)
MAX_QUERY_LENGTH = settings.get('query.max_query_length', 1000)
MAX_FILE_SIZE = settings.get('security.max_file_size', 10485760)  # 10MB
ALLOWED_FILE_TYPES = set(settings.get('security.allowed_file_types', ['pdf', 'txt', 'doc', 'docx']))
SENSITIVE_PATTERNS = settings.get('security.sensitive_patterns', [
    r"\b\d{16}\b",  # Credit card numbers
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email addresses
])

class SecurityError(Exception):
    """Base class for security-related errors."""
    pass

class InputValidationError(SecurityError):
    """Error raised when input validation fails."""
    pass

class RateLimitError(SecurityError):
    """Error raised when rate limit is exceeded."""
    pass

class TokenError(SecurityError):
    """Error raised when token validation fails."""
    pass

class FileValidationError(SecurityError):
    """Error raised when file validation fails."""
    pass

class SecurityManager:
    """Manages security-related functionality."""
    
    def __init__(self):
        """Initialize security manager."""
        self.rate_limits = {}
        self.blocked_ips = {}
        self.failed_attempts = {}
    
    def validate_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate input text for security concerns.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text:
            return False, "Input cannot be empty"
            
        if len(text) > MAX_QUERY_LENGTH:
            return False, f"Input exceeds maximum length of {MAX_QUERY_LENGTH} characters"
            
        # Check for sensitive patterns
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text):
                return False, "Input contains sensitive information"
                
        return True, None
    
    def validate_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file for security concerns.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                return False, f"File size exceeds maximum of {MAX_FILE_SIZE} bytes"
                
            # Check file type
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if file_ext not in ALLOWED_FILE_TYPES:
                return False, f"File type {file_ext} not allowed"
                
            return True, None
            
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    def check_rate_limit(self, ip: str) -> Tuple[bool, Optional[str]]:
        """
        Check if IP has exceeded rate limits.
        
        Args:
            ip: IP address to check
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()
        
        # Initialize rate limit tracking for IP
        if ip not in self.rate_limits:
            self.rate_limits[ip] = {
                'minute': {'count': 0, 'reset_time': current_time + 60},
                'hour': {'count': 0, 'reset_time': current_time + 3600}
            }
        
        # Check minute limit
        if current_time > self.rate_limits[ip]['minute']['reset_time']:
            self.rate_limits[ip]['minute'] = {'count': 0, 'reset_time': current_time + 60}
        elif self.rate_limits[ip]['minute']['count'] >= MAX_REQUESTS_PER_MINUTE:
            return False, "Rate limit exceeded: too many requests per minute"
        
        # Check hour limit
        if current_time > self.rate_limits[ip]['hour']['reset_time']:
            self.rate_limits[ip]['hour'] = {'count': 0, 'reset_time': current_time + 3600}
        elif self.rate_limits[ip]['hour']['count'] >= MAX_REQUESTS_PER_HOUR:
            return False, "Rate limit exceeded: too many requests per hour"
        
        # Update counters
        self.rate_limits[ip]['minute']['count'] += 1
        self.rate_limits[ip]['hour']['count'] += 1
        
        return True, None
    
    def generate_token(self, user_id: str, is_refresh: bool = False) -> str:
        """
        Generate JWT token.
        
        Args:
            user_id: User ID to encode in token
            is_refresh: Whether to generate a refresh token
            
        Returns:
            JWT token string
        """
        expiry = timedelta(days=settings.refresh_token_expire_days if is_refresh else 0,
                         minutes=settings.access_token_expire_minutes if not is_refresh else 0)
        
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + expiry,
            'type': 'refresh' if is_refresh else 'access'
        }
        
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """
        Verify JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Tuple of (is_valid, user_id)
        """
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
            return True, payload['user_id']
        except jwt.ExpiredSignatureError:
            return False, "Token has expired"
        except jwt.InvalidTokenError:
            return False, "Invalid token"
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Password to verify
            hashed: Hashed password to compare against
            
        Returns:
            True if password matches hash
        """
        return self.hash_password(password) == hashed

def rate_limit(func):
    """Decorator for rate limiting function calls."""
    @wraps(func)
    @sleep_and_retry
    @limits(calls=MAX_REQUESTS_PER_MINUTE, period=60)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def require_auth(func):
    """Decorator for requiring authentication."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        token = kwargs.get('token')
        if not token:
            raise ValueError("Authentication token required")
        
        security_manager = SecurityManager()
        is_valid, payload = security_manager.verify_token(token)
        
        if not is_valid:
            raise ValueError("Invalid or expired token")
        
        kwargs['user'] = payload
        return func(*args, **kwargs)
    return wrapper

def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security-related events."""
    logger.warning(f"Security Event - {event_type}: {json.dumps(details)}")

def check_ip_reputation(ip: str) -> Tuple[bool, Optional[str]]:
    """
    Check IP reputation using external service.
    Returns (is_safe, error_message)
    """
    try:
        response = requests.get(
            f"https://api.abuseipdb.com/api/v2/check",
            params={'ipAddress': ip},
            headers={'Key': os.getenv('ABUSEIPDB_KEY')},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data', {}).get('abuseConfidenceScore', 0) > 25:
                return False, "IP has poor reputation"
            return True, None
        else:
            return False, f"Error checking IP reputation: {response.status_code}"
    
    except requests.RequestException as e:
        logger.error(f"Error checking IP reputation: {str(e)}")
        return False, f"Error checking IP reputation: {str(e)}"

def validate_api_key(api_key: str) -> bool:
    """Validate API key format and checksum."""
    if not api_key or len(api_key) != 32:
        return False
    
    # Check if API key is in allowed list
    allowed_keys = os.getenv('ALLOWED_API_KEYS', '').split(',')
    return api_key in allowed_keys

def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data using Fernet symmetric encryption."""
    from cryptography.fernet import Fernet
    
    key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data using Fernet symmetric encryption."""
    from cryptography.fernet import Fernet
    
    key = os.getenv('ENCRYPTION_KEY')
    if not key:
        raise ValueError("Encryption key not found")
    
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

def validate_file_content(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate file content for security concerns.
    Returns (is_valid, error_message)
    """
    try:
        with open(file_path, 'rb') as f:
            # Check for binary content in text files
            if file_path.endswith('.txt'):
                content = f.read()
                if b'\x00' in content:
                    return False, "File contains binary content"
            
            # Check for malicious content
            content = f.read()
            malicious_patterns = [
                b'<?php',
                b'<script',
                b'eval(',
                b'exec(',
                b'system(',
                b'shell_exec(',
            ]
            
            for pattern in malicious_patterns:
                if pattern in content:
                    return False, f"File contains potentially malicious content: {pattern.decode()}"
        
        return True, None
    
    except Exception as e:
        return False, f"Error validating file content: {str(e)}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other security issues."""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove non-alphanumeric characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    
    # Ensure extension is allowed
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_FILE_TYPES:
        filename = f"{filename}.txt"
    
    return filename

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL for security concerns.
    Returns (is_valid, error_message)
    """
    try:
        # Check URL format
        if not re.match(r'^https?://', url):
            return False, "URL must start with http:// or https://"
        
        # Check for localhost or private IPs
        if re.match(r'^https?://(localhost|127\.0\.0\.1|192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)', url):
            return False, "URL cannot point to localhost or private IP"
        
        # Check URL length
        if len(url) > 2000:
            return False, "URL exceeds maximum length"
        
        return True, None
    
    except Exception as e:
        return False, f"Error validating URL: {str(e)}"

def check_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """
    Check password strength.
    Returns (is_strong, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, None 