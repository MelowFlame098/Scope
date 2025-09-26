"""Logging configuration for FinScope application.

This module provides:
- Structured logging with JSON format
- Multiple log handlers (console, file, rotating)
- Request/response logging
- Performance logging
- Security event logging
- Log filtering and formatting
"""

import logging
import logging.config
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback
import uuid
from contextlib import contextmanager

from app.config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        """Initialize JSON formatter.
        
        Args:
            include_extra: Whether to include extra fields
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'getMessage'
                }:
                    try:
                        # Only include JSON serializable values
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, ensure_ascii=False)


class RequestFormatter(logging.Formatter):
    """Formatter for HTTP request/response logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format request log record.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "request",
            "level": record.levelname,
            "message": record.getMessage()
        }
        
        # Add request-specific fields
        for field in ['request_id', 'method', 'url', 'status_code', 
                     'response_time', 'user_id', 'ip_address', 'user_agent']:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, ensure_ascii=False)


class SecurityFormatter(logging.Formatter):
    """Formatter for security event logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format security log record.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "security",
            "level": record.levelname,
            "event": record.getMessage(),
            "severity": getattr(record, 'severity', 'medium')
        }
        
        # Add security-specific fields
        for field in ['user_id', 'ip_address', 'user_agent', 'action', 
                     'resource', 'outcome', 'risk_score']:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, ensure_ascii=False)


class PerformanceFormatter(logging.Formatter):
    """Formatter for performance logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance log record.
        
        Args:
            record: Log record
            
        Returns:
            Formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "performance",
            "level": record.levelname,
            "operation": record.getMessage()
        }
        
        # Add performance-specific fields
        for field in ['duration', 'memory_usage', 'cpu_usage', 'db_queries',
                     'cache_hits', 'cache_misses', 'operation_type']:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        return json.dumps(log_data, ensure_ascii=False)


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from logs."""
    
    def __init__(self):
        """Initialize sensitive data filter."""
        super().__init__()
        self.sensitive_patterns = [
            'password', 'token', 'secret', 'key', 'authorization',
            'credit_card', 'ssn', 'social_security', 'api_key'
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to remove sensitive data.
        
        Args:
            record: Log record
            
        Returns:
            True to keep the record
        """
        # Check message
        message = record.getMessage().lower()
        for pattern in self.sensitive_patterns:
            if pattern in message:
                record.msg = "[SENSITIVE DATA REDACTED]"
                record.args = ()
                break
        
        # Check extra fields
        for key in list(record.__dict__.keys()):
            if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                setattr(record, key, "[REDACTED]")
        
        return True


class LoggingConfig:
    """Centralized logging configuration."""
    
    def __init__(self):
        """Initialize logging configuration."""
        self.settings = get_settings()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def get_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary.
        
        Returns:
            Logging configuration
        """
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "app.core.logging_config.JSONFormatter",
                    "include_extra": True
                },
                "simple": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
                },
                "request": {
                    "()": "app.core.logging_config.RequestFormatter"
                },
                "security": {
                    "()": "app.core.logging_config.SecurityFormatter"
                },
                "performance": {
                    "()": "app.core.logging_config.PerformanceFormatter"
                }
            },
            "filters": {
                "sensitive_data": {
                    "()": "app.core.logging_config.SensitiveDataFilter"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO" if self.settings.environment == "production" else "DEBUG",
                    "formatter": "json" if self.settings.environment == "production" else "detailed",
                    "stream": "ext://sys.stdout",
                    "filters": ["sensitive_data"]
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": str(self.log_dir / "finscope.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "filters": ["sensitive_data"]
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "json",
                    "filename": str(self.log_dir / "errors.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10,
                    "filters": ["sensitive_data"]
                },
                "request_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "request",
                    "filename": str(self.log_dir / "requests.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "security_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "WARNING",
                    "formatter": "security",
                    "filename": str(self.log_dir / "security.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10
                },
                "performance_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "performance",
                    "filename": str(self.log_dir / "performance.log"),
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                "finscope": {
                    "level": "DEBUG" if self.settings.environment == "development" else "INFO",
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "finscope.requests": {
                    "level": "INFO",
                    "handlers": ["request_file"],
                    "propagate": False
                },
                "finscope.security": {
                    "level": "WARNING",
                    "handlers": ["security_file", "console"],
                    "propagate": False
                },
                "finscope.performance": {
                    "level": "INFO",
                    "handlers": ["performance_file"],
                    "propagate": False
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["request_file"],
                    "propagate": False
                },
                "sqlalchemy": {
                    "level": "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "sqlalchemy.engine": {
                    "level": "INFO" if self.settings.environment == "development" else "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console", "file"]
            }
        }
        
        return config
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        config = self.get_config()
        logging.config.dictConfig(config)
        
        # Set up custom loggers
        self.setup_custom_loggers()
    
    def setup_custom_loggers(self) -> None:
        """Setup custom loggers for specific purposes."""
        # Request logger
        self.request_logger = logging.getLogger("finscope.requests")
        
        # Security logger
        self.security_logger = logging.getLogger("finscope.security")
        
        # Performance logger
        self.performance_logger = logging.getLogger("finscope.performance")
        
        # Main application logger
        self.app_logger = logging.getLogger("finscope")


class LoggerMixin:
    """Mixin to add logging capabilities to classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class.
        
        Returns:
            Logger instance
        """
        return logging.getLogger(f"finscope.{self.__class__.__name__}")


class RequestLogger:
    """Logger for HTTP requests and responses."""
    
    def __init__(self):
        """Initialize request logger."""
        self.logger = logging.getLogger("finscope.requests")
    
    def log_request(
        self,
        request_id: str,
        method: str,
        url: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log HTTP request.
        
        Args:
            request_id: Unique request ID
            method: HTTP method
            url: Request URL
            user_id: User ID if authenticated
            ip_address: Client IP address
            user_agent: User agent string
        """
        self.logger.info(
            "HTTP Request",
            extra={
                "request_id": request_id,
                "method": method,
                "url": url,
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        )
    
    def log_response(
        self,
        request_id: str,
        status_code: int,
        response_time: float,
        response_size: Optional[int] = None
    ) -> None:
        """Log HTTP response.
        
        Args:
            request_id: Unique request ID
            status_code: HTTP status code
            response_time: Response time in milliseconds
            response_size: Response size in bytes
        """
        self.logger.info(
            "HTTP Response",
            extra={
                "request_id": request_id,
                "status_code": status_code,
                "response_time": response_time,
                "response_size": response_size
            }
        )


class SecurityLogger:
    """Logger for security events."""
    
    def __init__(self):
        """Initialize security logger."""
        self.logger = logging.getLogger("finscope.security")
    
    def log_authentication_attempt(
        self,
        user_id: Optional[str],
        ip_address: str,
        success: bool,
        reason: Optional[str] = None
    ) -> None:
        """Log authentication attempt.
        
        Args:
            user_id: User ID
            ip_address: Client IP address
            success: Whether authentication succeeded
            reason: Failure reason if applicable
        """
        level = logging.INFO if success else logging.WARNING
        message = "Authentication succeeded" if success else f"Authentication failed: {reason}"
        
        self.logger.log(
            level,
            message,
            extra={
                "user_id": user_id,
                "ip_address": ip_address,
                "action": "authentication",
                "outcome": "success" if success else "failure",
                "severity": "low" if success else "medium"
            }
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: str
    ) -> None:
        """Log authorization failure.
        
        Args:
            user_id: User ID
            resource: Requested resource
            action: Attempted action
            ip_address: Client IP address
        """
        self.logger.warning(
            "Authorization denied",
            extra={
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "ip_address": ip_address,
                "outcome": "denied",
                "severity": "medium"
            }
        )
    
    def log_suspicious_activity(
        self,
        user_id: Optional[str],
        ip_address: str,
        activity: str,
        risk_score: int
    ) -> None:
        """Log suspicious activity.
        
        Args:
            user_id: User ID
            ip_address: Client IP address
            activity: Description of suspicious activity
            risk_score: Risk score (1-10)
        """
        severity = "high" if risk_score >= 8 else "medium" if risk_score >= 5 else "low"
        
        self.logger.error(
            f"Suspicious activity detected: {activity}",
            extra={
                "user_id": user_id,
                "ip_address": ip_address,
                "action": "suspicious_activity",
                "risk_score": risk_score,
                "severity": severity
            }
        )


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self):
        """Initialize performance logger."""
        self.logger = logging.getLogger("finscope.performance")
    
    def log_operation(
        self,
        operation: str,
        duration: float,
        operation_type: str = "general",
        **kwargs
    ) -> None:
        """Log operation performance.
        
        Args:
            operation: Operation name
            duration: Duration in milliseconds
            operation_type: Type of operation
            **kwargs: Additional metrics
        """
        extra = {
            "operation_type": operation_type,
            "duration": duration,
            **kwargs
        }
        
        self.logger.info(operation, extra=extra)
    
    @contextmanager
    def measure_operation(self, operation: str, operation_type: str = "general", **kwargs):
        """Context manager to measure operation duration.
        
        Args:
            operation: Operation name
            operation_type: Type of operation
            **kwargs: Additional metrics
        """
        start_time = datetime.utcnow()
        try:
            yield
        finally:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds() * 1000
            self.log_operation(operation, duration, operation_type, **kwargs)


# Global instances
logging_config = LoggingConfig()
request_logger = RequestLogger()
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()


def setup_logging() -> None:
    """Setup application logging."""
    logging_config.setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Get logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"finscope.{name}")


def generate_request_id() -> str:
    """Generate unique request ID.
    
    Returns:
        Unique request ID
    """
    return str(uuid.uuid4())