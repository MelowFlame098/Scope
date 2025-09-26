import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

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

class CryptoQuantMetrics:
    """Advanced on-chain quantitative metrics"""
    
    def calculate_mvrv_ratio(self, 
                            market_cap: float,
                            realized_cap: float) -> CryptoIndicatorResult:
        """Market Value to Realized Value ratio"""
        try:
            if realized_cap <= 0:
                raise ValueError("Realized cap must be positive")
                
            mvrv = market_cap / realized_cap
            
            # MVRV signals based on historical levels
            if mvrv < 1.0:  # Market cap below realized cap - potential bottom
                signal = 'buy'
                strength = 1.0 - mvrv
            elif mvrv > 3.7:  # Historical top territory
                signal = 'sell'
                strength = min((mvrv - 3.7) / 2.0, 1.0)
            else:
                signal = 'hold'
                strength = 0.5
                
            confidence = 0.9  # MVRV is a reliable indicator
            
            return CryptoIndicatorResult(
                indicator_name='MVRV Ratio',
                value=mvrv,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'market_cap': market_cap,
                    'realized_cap': realized_cap
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating MVRV: {e}")
            return self._error_result('MVRV Ratio', str(e))
    
    def calculate_sopr(self, 
                      spent_outputs_profit: float,
                      spent_outputs_loss: float) -> CryptoIndicatorResult:
        """Spent Output Profit Ratio"""
        try:
            total_spent = spent_outputs_profit + spent_outputs_loss
            if total_spent <= 0:
                raise ValueError("Total spent outputs must be positive")
                
            sopr = spent_outputs_profit / total_spent
            
            # SOPR signals
            if sopr < 0.95:  # Selling at a loss - potential bottom
                signal = 'buy'
                strength = (0.95 - sopr) / 0.1
            elif sopr > 1.05:  # Selling at profit - potential top
                signal = 'sell'
                strength = (sopr - 1.05) / 0.1
            else:
                signal = 'hold'
                strength = 0.5
                
            confidence = 0.85
            
            return CryptoIndicatorResult(
                indicator_name='SOPR',
                value=sopr,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'spent_outputs_profit': spent_outputs_profit,
                    'spent_outputs_loss': spent_outputs_loss,
                    'total_spent': total_spent
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating SOPR: {e}")
            return self._error_result('SOPR', str(e))
    
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