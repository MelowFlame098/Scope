from typing import Dict, List, Type
from .base_model import BaseModel
from .crypto_models import (
    StockToFlowModel, MetcalfeModel, NVTModel, 
    CryptoFinBERTModel, CryptoRLModel
)
from .stock_models import (
    DCFModel, CAPMModel, StockLSTMModel, StockXGBoostModel
)
from .forex_models import (
    PPPModel, ForexLSTMModel, IRPModel, ForexGARCHModel
)
from .futures_models import (
    CostOfCarryModel, ConvenienceYieldModel, 
    SamuelsonEffectModel, FuturesRLModel
)
from .index_models import (
    APTModel, DDMModel, KalmanFilterModel, 
    VECMModel, ElliottWaveModel
)
from .cross_asset_models import (
    ARIMAModel, GARCHModel, TransformerModel, 
    LightGBMModel, RSIMomentumModel, MACDModel, 
    IchimokuModel, PPORLModel, MarkowitzMPTModel
)


class ModelFactory:
    """
    Factory class for creating and managing trading models.
    """
    
    def __init__(self):
        self._model_registry: Dict[str, Type[BaseModel]] = {
            # Crypto Models
            'stock_to_flow': StockToFlowModel,
            'metcalfe': MetcalfeModel,
            'nvt': NVTModel,
            'crypto_finbert': CryptoFinBERTModel,
            'crypto_rl': CryptoRLModel,
            
            # Stock Models
            'dcf': DCFModel,
            'capm': CAPMModel,
            'stock_lstm': StockLSTMModel,
            'stock_xgboost': StockXGBoostModel,
            
            # Forex Models
            'ppp': PPPModel,
            'forex_lstm': ForexLSTMModel,
            'irp': IRPModel,
            'forex_garch': ForexGARCHModel,
            
            # Futures Models
            'cost_of_carry': CostOfCarryModel,
            'convenience_yield': ConvenienceYieldModel,
            'samuelson_effect': SamuelsonEffectModel,
            'futures_rl': FuturesRLModel,
            
            # Index Models
            'apt': APTModel,
            'ddm': DDMModel,
            'kalman_filter': KalmanFilterModel,
            'vecm': VECMModel,
            'elliott_wave': ElliottWaveModel,
            
            # Cross-Asset Models
            'arima': ARIMAModel,
            'garch': GARCHModel,
            'transformer': TransformerModel,
            'lightgbm': LightGBMModel,
            'rsi_momentum': RSIMomentumModel,
            'macd': MACDModel,
            'ichimoku': IchimokuModel,
            'ppo_rl': PPORLModel,
            'markowitz_mpt': MarkowitzMPTModel,
        }
        
        self._model_instances: Dict[str, BaseModel] = {}
    
    def create_model(self, model_id: str) -> BaseModel:
        """
        Create a model instance by ID.
        
        Args:
            model_id: The ID of the model to create
            
        Returns:
            BaseModel: Instance of the requested model
            
        Raises:
            ValueError: If model_id is not found in registry
        """
        if model_id not in self._model_registry:
            raise ValueError(f"Model '{model_id}' not found in registry")
        
        if model_id not in self._model_instances:
            model_class = self._model_registry[model_id]
            self._model_instances[model_id] = model_class()
        
        return self._model_instances[model_id]
    
    def get_model(self, model_id: str) -> BaseModel:
        """
        Get an existing model instance.
        
        Args:
            model_id: The ID of the model to get
            
        Returns:
            BaseModel: Instance of the requested model
            
        Raises:
            ValueError: If model_id is not found or not instantiated
        """
        if model_id not in self._model_instances:
            return self.create_model(model_id)
        
        return self._model_instances[model_id]
    
    def get_all_models(self) -> List[BaseModel]:
        """
        Get all available model instances.
        
        Returns:
            List[BaseModel]: List of all model instances
        """
        models = []
        for model_id in self._model_registry.keys():
            models.append(self.get_model(model_id))
        return models
    
    def get_models_by_category(self, category: str) -> List[BaseModel]:
        """
        Get all models in a specific category.
        
        Args:
            category: The category to filter by (e.g., 'Crypto', 'Stock', etc.)
            
        Returns:
            List[BaseModel]: List of models in the specified category
        """
        models = []
        for model in self.get_all_models():
            # Handle both enum and string categories
            model_category = model.category.value if hasattr(model.category, 'value') else str(model.category)
            if model_category.lower() == category.lower():
                models.append(model)
        return models
    
    def get_models_by_type(self, model_type: str) -> List[BaseModel]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: The model type to filter by
            
        Returns:
            List[BaseModel]: List of models of the specified type
        """
        models = []
        for model in self.get_all_models():
            # Handle both enum and string model types
            model_type_value = model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type)
            if model_type_value.lower() == model_type.lower():
                models.append(model)
        return models
    
    def get_available_model_ids(self) -> List[str]:
        """
        Get list of all available model IDs.
        
        Returns:
            List[str]: List of model IDs
        """
        return list(self._model_registry.keys())
    
    def get_model_info(self, model_id: str) -> Dict[str, str]:
        """
        Get information about a model without instantiating it.
        
        Args:
            model_id: The ID of the model
            
        Returns:
            Dict[str, str]: Model information
        """
        if model_id not in self._model_registry:
            raise ValueError(f"Model '{model_id}' not found in registry")
        
        # Create temporary instance to get info
        temp_model = self._model_registry[model_id]()
        return temp_model.get_info()
    
    def register_model(self, model_id: str, model_class: Type[BaseModel]):
        """
        Register a new model class.
        
        Args:
            model_id: The ID for the model
            model_class: The model class to register
        """
        self._model_registry[model_id] = model_class
    
    def unregister_model(self, model_id: str):
        """
        Unregister a model.
        
        Args:
            model_id: The ID of the model to unregister
        """
        if model_id in self._model_registry:
            del self._model_registry[model_id]
        
        if model_id in self._model_instances:
            del self._model_instances[model_id]


# Global model factory instance
model_factory = ModelFactory()