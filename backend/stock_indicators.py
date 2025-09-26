"""Stock Indicators Module - Main Orchestrator

This module orchestrates comprehensive stock analysis by coordinating
various specialized indicator models:
- DCF Model (Discounted Cash Flow)
- DDM Model (Dividend Discount Model)
- CAPM Model (Capital Asset Pricing Model)
- Fama-French Factor Models
- Advanced ML Models
- Financial Ratios Analysis

Author: Assistant
Date: 2024
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from indicators.stock.dcf_model import DCFModel, DCFResult
from indicators.stock.ddm_model import DividendDiscountModel, DDMResult
from indicators.stock.capm_model import CAPMModel, CAPMResult
from indicators.stock.fama_french_model import FamaFrenchModel, FamaFrenchResult
from indicators.stock.ml_models import AdvancedMLModels, MLModelResult
from indicators.stock.financial_ratios import FinancialRatiosCalculator, FinancialRatiosResult

class StockIndicatorType(Enum):
    """Stock-specific indicator types"""
    DCF = "dcf"  # Discounted Cash Flow
    DDM = "ddm"  # Dividend Discount Model
    CAPM = "capm"  # Capital Asset Pricing Model
    FAMA_FRENCH = "fama_french"  # Fama-French 3/5 Factor Model
    GORDON_GROWTH = "gordon_growth"  # Gordon Growth Model
    ARIMA = "arima"  # AutoRegressive Integrated Moving Average
    GARCH = "garch"  # Generalized AutoRegressive Conditional Heteroskedasticity
    VAR = "var"  # Vector AutoRegression
    KALMAN_FILTER = "kalman_filter"  # Kalman Filter
    LSTM = "lstm"  # Long Short-Term Memory
    XGBOOST = "xgboost"  # XGBoost
    BAYESIAN_NN = "bayesian_nn"  # Bayesian Neural Network
    AUTOML = "automl"  # Automated Machine Learning
    PEG_RATIO = "peg_ratio"  # Price/Earnings to Growth
    PRICE_TO_BOOK = "price_to_book"  # Price-to-Book Ratio
    ROE = "roe"  # Return on Equity
    DEBT_TO_EQUITY = "debt_to_equity"  # Debt-to-Equity Ratio
    EARNINGS_YIELD = "earnings_yield"  # Earnings Yield
    FREE_CASH_FLOW_YIELD = "fcf_yield"  # Free Cash Flow Yield

# Import shared data classes from modular components
from indicators.stock.dcf_model import StockFundamentals, MarketData

@dataclass
class StockIndicatorResult:
    """Result of stock indicator calculation"""
    indicator_type: StockIndicatorType
    value: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str = "HOLD"
    time_horizon: str = "medium_term"
    asset_symbols: List[str] = None

# All model classes have been moved to separate files in indicators/stock/
# This file now serves as the main orchestrator

class StockIndicatorEngine:
    """Main engine for calculating stock indicators using modular components"""
    
    def __init__(self):
        """Initialize the stock indicator engine with all modular components"""
        self.dcf_model = DCFModel()
        self.ddm_model = DividendDiscountModel()
        self.capm_model = CAPMModel()
        self.fama_french_model = FamaFrenchModel()
        self.ml_models = AdvancedMLModels()
        self.financial_ratios = FinancialRatiosCalculator()
        
    def get_available_indicators(self) -> List[StockIndicatorType]:
        """Get list of all available stock indicators"""
        return list(StockIndicatorType)
    
    def calculate_indicator(self, indicator_type: StockIndicatorType, 
                          fundamentals: StockFundamentals, 
                          market_data: MarketData,
                          price_data: Optional[List[float]] = None) -> StockIndicatorResult:
        """Calculate a specific stock indicator"""
        if indicator_type == StockIndicatorType.DCF:
            dcf_result = self.dcf_model.calculate(fundamentals, market_data)
            return self._convert_dcf_result(dcf_result)
        elif indicator_type == StockIndicatorType.DDM:
            ddm_result = self.ddm_model.calculate(fundamentals, market_data)
            return self._convert_ddm_result(ddm_result)
        elif indicator_type == StockIndicatorType.CAPM:
            capm_result = self.capm_model.calculate(fundamentals, market_data)
            return self._convert_capm_result(capm_result)
        elif indicator_type == StockIndicatorType.FAMA_FRENCH:
            ff_result = self.fama_french_model.calculate(fundamentals, market_data)
            return self._convert_fama_french_result(ff_result)
        elif indicator_type == StockIndicatorType.LSTM and price_data:
            ml_result = self.ml_models.lstm_prediction(price_data, fundamentals)
            return self._convert_ml_result(ml_result, StockIndicatorType.LSTM)
        elif indicator_type == StockIndicatorType.XGBOOST:
            xgb_result = self.ml_models.xgboost_analysis(fundamentals, market_data)
            return self._convert_ml_result(xgb_result, StockIndicatorType.XGBOOST)
        else:
            # Handle financial ratios
            ratios_result = self.financial_ratios.calculate_all_ratios(fundamentals, market_data)
            ratios_dict = self._convert_ratios_result(ratios_result)
            if indicator_type in ratios_dict:
                return ratios_dict[indicator_type]
            else:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")
    
    def calculate_all_indicators(self, fundamentals: StockFundamentals, 
                               market_data: MarketData, 
                               price_data: Optional[List[float]] = None) -> Dict[StockIndicatorType, StockIndicatorResult]:
        """Calculate all stock indicators using modular components"""
        results = {}
        
        # Valuation models
        dcf_result = self.dcf_model.calculate(fundamentals, market_data)
        results[StockIndicatorType.DCF] = self._convert_dcf_result(dcf_result)
        
        ddm_result = self.ddm_model.calculate(fundamentals, market_data)
        results[StockIndicatorType.DDM] = self._convert_ddm_result(ddm_result)
        
        capm_result = self.capm_model.calculate(fundamentals, market_data)
        results[StockIndicatorType.CAPM] = self._convert_capm_result(capm_result)
        
        fama_french_result = self.fama_french_model.calculate(fundamentals, market_data)
        results[StockIndicatorType.FAMA_FRENCH] = self._convert_fama_french_result(fama_french_result)
        
        # ML models
        if price_data:
            ml_result = self.ml_models.lstm_prediction(price_data, fundamentals)
            results[StockIndicatorType.LSTM] = self._convert_ml_result(ml_result, StockIndicatorType.LSTM)
        
        xgb_result = self.ml_models.xgboost_analysis(fundamentals, market_data)
        results[StockIndicatorType.XGBOOST] = self._convert_ml_result(xgb_result, StockIndicatorType.XGBOOST)
        
        # Financial ratios
        ratios_result = self.financial_ratios.calculate_all_ratios(fundamentals, market_data)
        results.update(self._convert_ratios_result(ratios_result))
        
        return results
    
    def _convert_dcf_result(self, dcf_result: DCFResult) -> StockIndicatorResult:
        """Convert DCF result to StockIndicatorResult"""
        return StockIndicatorResult(
            indicator_type=StockIndicatorType.DCF,
            value=dcf_result.intrinsic_value,
            confidence=dcf_result.confidence,
            metadata=dcf_result.metadata,
            timestamp=dcf_result.timestamp,
            interpretation=dcf_result.interpretation,
            risk_level=dcf_result.risk_level
        )
    
    def _convert_ddm_result(self, ddm_result: DDMResult) -> StockIndicatorResult:
        """Convert DDM result to StockIndicatorResult"""
        return StockIndicatorResult(
            indicator_type=StockIndicatorType.DDM,
            value=ddm_result.intrinsic_value,
            confidence=ddm_result.confidence,
            metadata=ddm_result.metadata,
            timestamp=ddm_result.timestamp,
            interpretation=ddm_result.interpretation,
            risk_level=ddm_result.risk_level
        )
    
    def _convert_capm_result(self, capm_result: CAPMResult) -> StockIndicatorResult:
        """Convert CAPM result to StockIndicatorResult"""
        return StockIndicatorResult(
            indicator_type=StockIndicatorType.CAPM,
            value=capm_result.expected_return,
            confidence=capm_result.confidence,
            metadata=capm_result.metadata,
            timestamp=capm_result.timestamp,
            interpretation=capm_result.interpretation,
            risk_level=capm_result.risk_level
        )
    
    def _convert_fama_french_result(self, ff_result: FamaFrenchResult) -> StockIndicatorResult:
        """Convert Fama-French result to StockIndicatorResult"""
        return StockIndicatorResult(
            indicator_type=StockIndicatorType.FAMA_FRENCH,
            value=ff_result.expected_return,
            confidence=ff_result.confidence,
            metadata=ff_result.metadata,
            timestamp=ff_result.timestamp,
            interpretation=ff_result.interpretation,
            risk_level=ff_result.risk_level
        )
    
    def _convert_ml_result(self, ml_result: MLModelResult, indicator_type: StockIndicatorType) -> StockIndicatorResult:
        """Convert ML result to StockIndicatorResult"""
        return StockIndicatorResult(
            indicator_type=indicator_type,
            value=ml_result.prediction,
            confidence=ml_result.confidence,
            metadata=ml_result.metadata,
            timestamp=ml_result.timestamp,
            interpretation=ml_result.interpretation,
            risk_level=ml_result.risk_level
        )
    
    def _convert_ratios_result(self, ratios_result: FinancialRatiosResult) -> Dict[StockIndicatorType, StockIndicatorResult]:
        """Convert financial ratios result to StockIndicatorResult dictionary"""
        results = {}
        
        # Map ratio names to indicator types
        ratio_mapping = {
            'peg_ratio': StockIndicatorType.PEG_RATIO,
            'price_to_book': StockIndicatorType.PRICE_TO_BOOK,
            'roe': StockIndicatorType.ROE,
            'debt_to_equity': StockIndicatorType.DEBT_TO_EQUITY,
            'earnings_yield': StockIndicatorType.EARNINGS_YIELD,
            'fcf_yield': StockIndicatorType.FREE_CASH_FLOW_YIELD
        }
        
        for ratio_name, indicator_type in ratio_mapping.items():
            if ratio_name in ratios_result.ratios:
                ratio_data = ratios_result.ratios[ratio_name]
                results[indicator_type] = StockIndicatorResult(
                    indicator_type=indicator_type,
                    value=ratio_data['value'],
                    confidence=ratio_data.get('confidence', 0.8),
                    metadata=ratio_data.get('metadata', {}),
                    timestamp=ratios_result.timestamp,
                    interpretation=ratio_data.get('interpretation', f"{ratio_name}: {ratio_data['value']:.2f}"),
                    risk_level=ratio_data.get('risk_level', 'medium')
                )
        
        return results