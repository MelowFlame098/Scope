import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CryptoIndicatorResult:
    """Result container for crypto indicator calculations"""
    indicator_name: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 signal strength

class OnChainMLModel:
    """Machine learning models for on-chain data analysis"""
    
    def __init__(self):
        self.rf_model = None
        self.mlp_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        
    def train_models(self, 
                    features: List[List[float]], 
                    targets: List[float]) -> bool:
        """Train ML models on blockchain data"""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available, using mock predictions")
                return False
                
            if len(features) != len(targets) or len(features) < 10:
                raise ValueError("Need at least 10 training samples")
                
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.rf_model.fit(X_train, y_train)
            
            # Train MLP
            self.mlp_model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
            self.mlp_model.fit(X_train, y_train)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def predict(self, 
               features: List[float],
               model_type: str = 'ensemble') -> CryptoIndicatorResult:
        """Make predictions using trained models"""
        try:
            if not SKLEARN_AVAILABLE or not self.is_trained:
                # Mock prediction
                prediction = np.random.uniform(0.3, 0.7)
                confidence = 0.5
            else:
                # Scale features
                X = np.array(features).reshape(1, -1)
                X_scaled = self.scaler.transform(X)
                
                if model_type == 'rf':
                    prediction = self.rf_model.predict(X_scaled)[0]
                    confidence = 0.8
                elif model_type == 'mlp':
                    prediction = self.mlp_model.predict(X_scaled)[0]
                    confidence = 0.75
                else:  # ensemble
                    rf_pred = self.rf_model.predict(X_scaled)[0]
                    mlp_pred = self.mlp_model.predict(X_scaled)[0]
                    prediction = (rf_pred + mlp_pred) / 2
                    confidence = 0.85
            
            # Generate trading signals
            if prediction > 0.6:
                signal = 'buy'
                strength = min(prediction, 1.0)
            elif prediction < 0.4:
                signal = 'sell'
                strength = min(1.0 - prediction, 1.0)
            else:
                signal = 'hold'
                strength = 0.5
            
            return CryptoIndicatorResult(
                indicator_name='OnChain ML Model',
                value=prediction,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'model_type': model_type,
                    'is_trained': self.is_trained,
                    'feature_count': len(features),
                    'sklearn_available': SKLEARN_AVAILABLE
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._error_result('OnChain ML Model', str(e))
    
    def _error_result(self, indicator_name: str, error_msg: str) -> CryptoIndicatorResult:
        return CryptoIndicatorResult(
            indicator_name=indicator_name,
            value=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_msg},
            signal='hold',
            strength=0.0
        )