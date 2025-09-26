from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from enum import Enum

class ModelType(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    ML = "ml"
    SENTIMENT = "sentiment"

class AssetCategory(Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    FUTURES = "futures"
    INDEXES = "indexes"
    CROSS_ASSET = "cross-asset"

class ModelStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class BaseModel(ABC):
    """
    Abstract base class for all trading models in FinScope.
    Provides common interface and functionality for model execution,
    prediction, and performance tracking.
    """
    
    def __init__(self, model_id: str, name: str, category: AssetCategory, 
                 model_type: ModelType, description: str):
        self.model_id = model_id
        self.name = name
        self.category = category
        self.model_type = model_type
        self.description = description
        self.status = ModelStatus.IDLE
        self.accuracy = 0.0
        self.last_run = None
        self.is_active = True
        self.logger = logging.getLogger(f"model.{model_id}")
        self.parameters = {}
        self.training_data = None
        self.model_instance = None
        
    @abstractmethod
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare and preprocess data for the model.
        
        Args:
            data: Raw market data
            **kwargs: Additional parameters for data preparation
            
        Returns:
            Preprocessed data ready for model training/prediction
        """
        pass
    
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train the model with provided data.
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate predictions using the trained model.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results with confidence intervals
        """
        pass
    
    @abstractmethod
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Performance metrics
        """
        pass
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: Dictionary of model parameters
        """
        self.parameters.update(parameters)
        self.logger.info(f"Updated parameters for {self.name}: {parameters}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information and current status.
        
        Returns:
            Model information dictionary
        """
        return {
            'id': self.model_id,
            'name': self.name,
            'category': self.category.value,
            'type': self.model_type.value,
            'description': self.description,
            'status': self.status.value,
            'accuracy': self.accuracy,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'is_active': self.is_active,
            'parameters': self.parameters
        }
    
    def start_run(self) -> None:
        """
        Mark the start of a model run.
        """
        self.status = ModelStatus.RUNNING
        self.last_run = datetime.now()
        self.logger.info(f"Started running {self.name}")
    
    def complete_run(self, accuracy: Optional[float] = None) -> None:
        """
        Mark the completion of a model run.
        
        Args:
            accuracy: Model accuracy if available
        """
        self.status = ModelStatus.COMPLETED
        if accuracy is not None:
            self.accuracy = accuracy
        self.logger.info(f"Completed running {self.name} with accuracy: {self.accuracy}")
    
    def error_run(self, error_message: str) -> None:
        """
        Mark a model run as failed.
        
        Args:
            error_message: Error description
        """
        self.status = ModelStatus.ERROR
        self.logger.error(f"Error in {self.name}: {error_message}")
    
    def reset_status(self) -> None:
        """
        Reset model status to idle.
        """
        self.status = ModelStatus.IDLE
        self.logger.info(f"Reset status for {self.name}")
    
    def calculate_common_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate common performance metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of performance metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Calculate directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate input data format and required columns.
        
        Args:
            data: Input data to validate
            required_columns: List of required column names
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.error("Data is None or empty")
            return False
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available for the model.
        
        Returns:
            Feature importance dictionary or None
        """
        return None
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import joblib
            model_data = {
                'model_instance': self.model_instance,
                'parameters': self.parameters,
                'accuracy': self.accuracy,
                'last_run': self.last_run
            }
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.model_instance = model_data['model_instance']
            self.parameters = model_data['parameters']
            self.accuracy = model_data['accuracy']
            self.last_run = model_data['last_run']
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False