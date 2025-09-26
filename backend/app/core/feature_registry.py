"""Feature Registry for FinScope.

This module provides a centralized system for managing optional features
and their dependencies. Features are loaded dynamically based on:
- Available dependencies
- Configuration settings
- Environment variables
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import logging
import importlib
from functools import wraps

logger = logging.getLogger(__name__)


class FeatureStatus(str, Enum):
    """Feature loading status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class FeatureInfo:
    """Information about a feature."""
    name: str
    description: str
    dependencies: List[str]
    status: FeatureStatus
    instance: Optional[Any] = None
    error_message: Optional[str] = None
    config_key: Optional[str] = None


class FeatureRegistry:
    """Registry for managing optional features."""
    
    def __init__(self):
        self.features: Dict[str, FeatureInfo] = {}
        self.loaders: Dict[str, Callable] = {}
        self._initialized = False
    
    def register_feature(
        self,
        name: str,
        description: str,
        dependencies: List[str],
        loader: Callable,
        config_key: Optional[str] = None
    ) -> None:
        """Register a feature with its loader function.
        
        Args:
            name: Feature name
            description: Feature description
            dependencies: List of required Python packages
            loader: Function that returns the feature instance
            config_key: Configuration key to check if feature is enabled
        """
        self.features[name] = FeatureInfo(
            name=name,
            description=description,
            dependencies=dependencies,
            status=FeatureStatus.UNAVAILABLE,
            config_key=config_key
        )
        self.loaders[name] = loader
        logger.debug(f"Registered feature: {name}")
    
    def check_dependencies(self, dependencies: List[str]) -> tuple[bool, List[str]]:
        """Check if all dependencies are available.
        
        Args:
            dependencies: List of package names to check
            
        Returns:
            Tuple of (all_available, missing_packages)
        """
        missing = []
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing.append(dep)
        
        return len(missing) == 0, missing
    
    def load_feature(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a specific feature.
        
        Args:
            name: Feature name
            config: Configuration dictionary
            
        Returns:
            True if feature loaded successfully, False otherwise
        """
        if name not in self.features:
            logger.error(f"Feature '{name}' not registered")
            return False
        
        feature_info = self.features[name]
        
        # Check if feature is disabled in config
        if config and feature_info.config_key:
            if not config.get(feature_info.config_key, True):
                feature_info.status = FeatureStatus.DISABLED
                logger.info(f"Feature '{name}' disabled in configuration")
                return False
        
        # Check dependencies
        deps_available, missing_deps = self.check_dependencies(feature_info.dependencies)
        if not deps_available:
            feature_info.status = FeatureStatus.UNAVAILABLE
            feature_info.error_message = f"Missing dependencies: {', '.join(missing_deps)}"
            logger.warning(f"Feature '{name}' unavailable: {feature_info.error_message}")
            return False
        
        # Load the feature
        try:
            loader = self.loaders[name]
            feature_info.instance = loader()
            feature_info.status = FeatureStatus.AVAILABLE
            logger.info(f"Feature '{name}' loaded successfully")
            return True
        except Exception as e:
            feature_info.status = FeatureStatus.ERROR
            feature_info.error_message = str(e)
            logger.error(f"Failed to load feature '{name}': {e}")
            return False
    
    def load_all_features(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Load all registered features.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary mapping feature names to load success status
        """
        results = {}
        for name in self.features:
            results[name] = self.load_feature(name, config)
        
        self._initialized = True
        return results
    
    def get_feature(self, name: str) -> Optional[Any]:
        """Get a loaded feature instance.
        
        Args:
            name: Feature name
            
        Returns:
            Feature instance if available, None otherwise
        """
        if name not in self.features:
            return None
        
        feature_info = self.features[name]
        if feature_info.status == FeatureStatus.AVAILABLE:
            return feature_info.instance
        
        return None
    
    def is_available(self, name: str) -> bool:
        """Check if a feature is available.
        
        Args:
            name: Feature name
            
        Returns:
            True if feature is available, False otherwise
        """
        if name not in self.features:
            return False
        
        return self.features[name].status == FeatureStatus.AVAILABLE
    
    def get_status(self, name: str) -> Optional[FeatureStatus]:
        """Get feature status.
        
        Args:
            name: Feature name
            
        Returns:
            Feature status if registered, None otherwise
        """
        if name not in self.features:
            return None
        
        return self.features[name].status
    
    def get_all_features(self) -> Dict[str, FeatureInfo]:
        """Get all registered features.
        
        Returns:
            Dictionary of all features
        """
        return self.features.copy()
    
    def get_available_features(self) -> Dict[str, Any]:
        """Get all available feature instances.
        
        Returns:
            Dictionary of available features
        """
        available = {}
        for name, info in self.features.items():
            if info.status == FeatureStatus.AVAILABLE and info.instance:
                available[name] = info.instance
        
        return available
    
    def get_feature_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all features.
        
        Returns:
            Dictionary with feature summaries
        """
        summary = {}
        for name, info in self.features.items():
            summary[name] = {
                "description": info.description,
                "status": info.status.value,
                "dependencies": info.dependencies,
                "error_message": info.error_message,
                "available": info.status == FeatureStatus.AVAILABLE
            }
        
        return summary


def requires_feature(feature_name: str):
    """Decorator to mark functions that require a specific feature.
    
    Args:
        feature_name: Name of the required feature
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not registry.is_available(feature_name):
                raise RuntimeError(
                    f"Feature '{feature_name}' is not available. "
                    f"Status: {registry.get_status(feature_name)}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def optional_feature(feature_name: str, default_return=None):
    """Decorator for functions that optionally use a feature.
    
    Args:
        feature_name: Name of the optional feature
        default_return: Value to return if feature is not available
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not registry.is_available(feature_name):
                logger.warning(
                    f"Feature '{feature_name}' not available, "
                    f"returning default: {default_return}"
                )
                return default_return
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global feature registry instance
registry = FeatureRegistry()


# Feature loader functions
def load_ai_core():
    """Load AI Core features."""
    from ai_core import AICore
    config = {
        'autonomous_trading': {
            'enabled': True,
            'risk_tolerance': 0.05,
            'max_position_size': 0.1
        },
        'natural_language': {
            'enabled': True,
            'model_name': 'gpt-4',
            'max_tokens': 2048
        }
    }
    return AICore(config=config)


def load_defi_core():
    """Load DeFi Core features."""
    from defi_core import DeFiCore
    config = {
        'protocols': {
            'uniswap': True,
            'compound': True,
            'aave': True
        },
        'networks': {
            'ethereum': True,
            'polygon': True,
            'bsc': False
        }
    }
    return DeFiCore(config=config)


def load_langchain_integration():
    """Load LangChain integration."""
    from langchain_integration import LangChainIntegration
    return LangChainIntegration()


def load_technical_analysis():
    """Load Technical Analysis service."""
    from technical_analysis import TechnicalAnalysisService
    return TechnicalAnalysisService()


def load_ml_pipeline():
    """Load ML Pipeline service."""
    from ml_pipeline import MLPipelineService
    return MLPipelineService()


def load_reinforcement_learning():
    """Load Reinforcement Learning service."""
    from reinforcement_learning import ReinforcementLearningService
    return ReinforcementLearningService()


# Register all features
def register_all_features():
    """Register all available features."""
    
    # AI Features
    registry.register_feature(
        name="ai_core",
        description="AI Core Engine with autonomous trading and NLP",
        dependencies=["openai", "transformers", "torch"],
        loader=load_ai_core,
        config_key="enable_ai_features"
    )
    
    registry.register_feature(
        name="langchain",
        description="LangChain integration for advanced LLM workflows",
        dependencies=["langchain", "langchain_openai"],
        loader=load_langchain_integration,
        config_key="enable_ai_features"
    )
    
    registry.register_feature(
        name="technical_analysis",
        description="Technical analysis with TA-Lib and pandas-ta",
        dependencies=["talib", "pandas_ta"],
        loader=load_technical_analysis
    )
    
    registry.register_feature(
        name="ml_pipeline",
        description="Machine learning pipeline with scikit-learn",
        dependencies=["sklearn", "xgboost", "lightgbm"],
        loader=load_ml_pipeline,
        config_key="enable_ai_features"
    )
    
    registry.register_feature(
        name="reinforcement_learning",
        description="Reinforcement learning for trading strategies",
        dependencies=["gym", "stable_baselines3"],
        loader=load_reinforcement_learning,
        config_key="enable_ai_features"
    )
    
    # DeFi Features
    registry.register_feature(
        name="defi_core",
        description="DeFi Core Engine with protocol integration",
        dependencies=["web3", "eth_account", "ccxt"],
        loader=load_defi_core,
        config_key="enable_defi_features"
    )
    
    logger.info("All features registered")


# Initialize features on module import
register_all_features()