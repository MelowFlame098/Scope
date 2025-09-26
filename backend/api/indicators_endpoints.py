"""Technical Indicators API Endpoints

Provides REST API endpoints for all technical indicators across different asset classes:
- Crypto indicators (S2F, Metcalfe's Law, NVT/NVM, etc.)
- Stock indicators (DCF, DDM, CAPM, Fama-French, etc.)
- Forex indicators (PPP, IRP, UIP, etc.)
- Futures indicators (Cost-of-Carry, Convenience Yield, etc.)
- Index indicators (Macro Factors, APT, etc.)
- Cross-asset indicators (ARIMA, GARCH, LSTM, etc.)
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import logging
import asyncio

# Import our indicator engines
try:
    from ..crypto_indicators import CryptoIndicatorEngine, CryptoIndicatorType, CryptoAssetData
    from ..stock_indicators import StockIndicatorEngine, StockIndicatorType, StockAssetData
    from ..forex_indicators import ForexIndicatorEngine, ForexIndicatorType, ForexAssetData
    from ..futures_indicators import FuturesIndicatorEngine, FuturesIndicatorType, FuturesAssetData
    from ..index_indicators import IndexIndicatorEngine, IndexIndicatorType, IndexAssetData
    from ..cross_asset_indicators import CrossAssetIndicatorEngine, CrossAssetIndicatorType, AssetData
except ImportError as e:
    logging.warning(f"Could not import indicator engines: {e}")
    # Fallback for development
    CryptoIndicatorEngine = None
    StockIndicatorEngine = None
    ForexIndicatorEngine = None
    FuturesIndicatorEngine = None
    IndexIndicatorEngine = None
    CrossAssetIndicatorEngine = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/indicators", tags=["Technical Indicators"])

# Initialize indicator engines
crypto_engine = CryptoIndicatorEngine() if CryptoIndicatorEngine else None
stock_engine = StockIndicatorEngine() if StockIndicatorEngine else None
forex_engine = ForexIndicatorEngine() if ForexIndicatorEngine else None
futures_engine = FuturesIndicatorEngine() if FuturesIndicatorEngine else None
index_engine = IndexIndicatorEngine() if IndexIndicatorEngine else None
cross_asset_engine = CrossAssetIndicatorEngine() if CrossAssetIndicatorEngine else None

# Pydantic models for request/response

class AssetDataRequest(BaseModel):
    """Base asset data request model"""
    symbol: str = Field(..., description="Asset symbol (e.g., BTC-USD, AAPL, EUR/USD)")
    asset_type: str = Field(..., description="Asset type: crypto, stock, forex, futures, index")
    current_price: float = Field(..., description="Current asset price")
    historical_prices: List[float] = Field(..., description="Historical price data")
    volume: Optional[float] = Field(None, description="Current volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    volatility: Optional[float] = Field(0.0, description="Historical volatility")
    beta: Optional[float] = Field(None, description="Beta coefficient")
    fundamental_data: Optional[Dict[str, Any]] = Field(None, description="Additional fundamental data")

class CryptoDataRequest(AssetDataRequest):
    """Crypto-specific data request"""
    circulating_supply: Optional[float] = Field(None, description="Circulating supply")
    total_supply: Optional[float] = Field(None, description="Total supply")
    network_value: Optional[float] = Field(None, description="Network value")
    transaction_volume: Optional[float] = Field(None, description="Transaction volume")
    active_addresses: Optional[int] = Field(None, description="Active addresses")
    hash_rate: Optional[float] = Field(None, description="Network hash rate")
    realized_cap: Optional[float] = Field(None, description="Realized capitalization")
    mvrv_ratio: Optional[float] = Field(None, description="MVRV ratio")
    nvt_ratio: Optional[float] = Field(None, description="NVT ratio")
    social_sentiment: Optional[float] = Field(None, description="Social sentiment score")

class StockDataRequest(AssetDataRequest):
    """Stock-specific data request"""
    earnings_per_share: Optional[float] = Field(None, description="Earnings per share")
    book_value_per_share: Optional[float] = Field(None, description="Book value per share")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    pe_ratio: Optional[float] = Field(None, description="Price-to-earnings ratio")
    pb_ratio: Optional[float] = Field(None, description="Price-to-book ratio")
    roe: Optional[float] = Field(None, description="Return on equity")
    debt_to_equity: Optional[float] = Field(None, description="Debt-to-equity ratio")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")
    free_cash_flow: Optional[float] = Field(None, description="Free cash flow")
    shares_outstanding: Optional[float] = Field(None, description="Shares outstanding")

class ForexDataRequest(AssetDataRequest):
    """Forex-specific data request"""
    base_currency: str = Field(..., description="Base currency (e.g., EUR)")
    quote_currency: str = Field(..., description="Quote currency (e.g., USD)")
    interest_rate_base: Optional[float] = Field(None, description="Base currency interest rate")
    interest_rate_quote: Optional[float] = Field(None, description="Quote currency interest rate")
    inflation_rate_base: Optional[float] = Field(None, description="Base currency inflation rate")
    inflation_rate_quote: Optional[float] = Field(None, description="Quote currency inflation rate")
    gdp_growth_base: Optional[float] = Field(None, description="Base currency GDP growth")
    gdp_growth_quote: Optional[float] = Field(None, description="Quote currency GDP growth")
    trade_balance: Optional[float] = Field(None, description="Trade balance")
    political_stability: Optional[float] = Field(None, description="Political stability index")

class FuturesDataRequest(AssetDataRequest):
    """Futures-specific data request"""
    underlying_asset: str = Field(..., description="Underlying asset symbol")
    expiration_date: datetime = Field(..., description="Contract expiration date")
    contract_size: float = Field(..., description="Contract size")
    spot_price: float = Field(..., description="Spot price of underlying")
    risk_free_rate: Optional[float] = Field(None, description="Risk-free interest rate")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield of underlying")
    storage_cost: Optional[float] = Field(None, description="Storage cost")
    convenience_yield: Optional[float] = Field(None, description="Convenience yield")
    open_interest: Optional[float] = Field(None, description="Open interest")
    basis: Optional[float] = Field(None, description="Basis (futures - spot)")

class IndexDataRequest(AssetDataRequest):
    """Index-specific data request"""
    constituent_symbols: List[str] = Field(..., description="Index constituent symbols")
    constituent_weights: List[float] = Field(..., description="Constituent weights")
    dividend_yield: Optional[float] = Field(None, description="Index dividend yield")
    pe_ratio: Optional[float] = Field(None, description="Index P/E ratio")
    sector_weights: Optional[Dict[str, float]] = Field(None, description="Sector weights")
    market_cap_weighted: Optional[bool] = Field(True, description="Is market cap weighted")
    rebalance_frequency: Optional[str] = Field(None, description="Rebalance frequency")
    expense_ratio: Optional[float] = Field(None, description="Expense ratio")

class IndicatorRequest(BaseModel):
    """Generic indicator calculation request"""
    asset_data: AssetDataRequest
    indicator_types: List[str] = Field(..., description="List of indicator types to calculate")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for indicators")
    additional_assets: Optional[List[AssetDataRequest]] = Field(None, description="Additional assets for cross-asset analysis")

class IndicatorResponse(BaseModel):
    """Indicator calculation response"""
    indicator_type: str
    value: Union[float, Dict[str, float], List[float]]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str
    time_horizon: str
    asset_symbols: List[str]

class BatchIndicatorResponse(BaseModel):
    """Batch indicator calculation response"""
    asset_symbol: str
    asset_type: str
    indicators: Dict[str, IndicatorResponse]
    calculation_time: float
    success: bool
    error_message: Optional[str] = None

# Helper functions

def convert_to_asset_data(request: AssetDataRequest) -> AssetData:
    """Convert request model to AssetData"""
    return AssetData(
        symbol=request.symbol,
        asset_type=request.asset_type,
        current_price=request.current_price,
        historical_prices=request.historical_prices,
        volume=request.volume or 0.0,
        market_cap=request.market_cap,
        volatility=request.volatility,
        beta=request.beta,
        fundamental_data=request.fundamental_data or {}
    )

def convert_to_crypto_data(request: CryptoDataRequest) -> 'CryptoAssetData':
    """Convert request model to CryptoAssetData"""
    if not CryptoAssetData:
        raise HTTPException(status_code=500, detail="Crypto indicators not available")
    
    return CryptoAssetData(
        symbol=request.symbol,
        current_price=request.current_price,
        historical_prices=request.historical_prices,
        volume=request.volume or 0.0,
        market_cap=request.market_cap or 0.0,
        circulating_supply=request.circulating_supply or 0.0,
        total_supply=request.total_supply or 0.0,
        network_value=request.network_value or 0.0,
        transaction_volume=request.transaction_volume or 0.0,
        active_addresses=request.active_addresses or 0,
        hash_rate=request.hash_rate or 0.0,
        realized_cap=request.realized_cap or 0.0,
        mvrv_ratio=request.mvrv_ratio or 1.0,
        nvt_ratio=request.nvt_ratio or 0.0,
        social_sentiment=request.social_sentiment or 0.0
    )

# API Endpoints

@router.get("/health")
async def health_check():
    """Health check for indicators service"""
    engines_status = {
        "crypto": crypto_engine is not None,
        "stock": stock_engine is not None,
        "forex": forex_engine is not None,
        "futures": futures_engine is not None,
        "index": index_engine is not None,
        "cross_asset": cross_asset_engine is not None
    }
    
    return {
        "status": "healthy",
        "engines": engines_status,
        "timestamp": datetime.now()
    }

@router.get("/types")
async def get_indicator_types():
    """Get all available indicator types by asset class"""
    try:
        indicator_types = {
            "crypto": [e.value for e in CryptoIndicatorType] if CryptoIndicatorType else [],
            "stock": [e.value for e in StockIndicatorType] if StockIndicatorType else [],
            "forex": [e.value for e in ForexIndicatorType] if ForexIndicatorType else [],
            "futures": [e.value for e in FuturesIndicatorType] if FuturesIndicatorType else [],
            "index": [e.value for e in IndexIndicatorType] if IndexIndicatorType else [],
            "cross_asset": [e.value for e in CrossAssetIndicatorType] if CrossAssetIndicatorType else []
        }
        
        return {
            "indicator_types": indicator_types,
            "total_indicators": sum(len(types) for types in indicator_types.values())
        }
    except Exception as e:
        logger.error(f"Error getting indicator types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crypto/calculate")
async def calculate_crypto_indicators(
    request: CryptoDataRequest,
    indicator_types: Optional[List[str]] = Query(None, description="Specific indicators to calculate")
):
    """Calculate crypto-specific indicators"""
    try:
        if not crypto_engine:
            raise HTTPException(status_code=500, detail="Crypto indicator engine not available")
        
        crypto_data = convert_to_crypto_data(request)
        
        start_time = datetime.now()
        
        if indicator_types:
            # Calculate specific indicators
            results = {}
            for indicator_type in indicator_types:
                try:
                    crypto_indicator_type = CryptoIndicatorType(indicator_type)
                    result = crypto_engine.calculate_indicator(crypto_data, crypto_indicator_type)
                    results[indicator_type] = IndicatorResponse(
                        indicator_type=result.indicator_type.value,
                        value=result.value,
                        confidence=result.confidence,
                        metadata=result.metadata,
                        timestamp=result.timestamp,
                        interpretation=result.interpretation,
                        risk_level=result.risk_level,
                        signal=result.signal,
                        time_horizon=result.time_horizon,
                        asset_symbols=result.asset_symbols
                    )
                except ValueError:
                    logger.warning(f"Unknown crypto indicator type: {indicator_type}")
                    continue
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type}: {str(e)}")
                    continue
        else:
            # Calculate all indicators
            all_results = crypto_engine.calculate_all_indicators(crypto_data)
            results = {}
            for indicator_type, result in all_results.items():
                results[indicator_type.value] = IndicatorResponse(
                    indicator_type=result.indicator_type.value,
                    value=result.value,
                    confidence=result.confidence,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    interpretation=result.interpretation,
                    risk_level=result.risk_level,
                    signal=result.signal,
                    time_horizon=result.time_horizon,
                    asset_symbols=result.asset_symbols
                )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="crypto",
            indicators=results,
            calculation_time=calculation_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating crypto indicators: {str(e)}")
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="crypto",
            indicators={},
            calculation_time=0.0,
            success=False,
            error_message=str(e)
        )

@router.post("/stock/calculate")
async def calculate_stock_indicators(
    request: StockDataRequest,
    indicator_types: Optional[List[str]] = Query(None, description="Specific indicators to calculate")
):
    """Calculate stock-specific indicators"""
    try:
        if not stock_engine:
            raise HTTPException(status_code=500, detail="Stock indicator engine not available")
        
        # Convert to StockAssetData (simplified conversion)
        stock_data = StockAssetData(
            symbol=request.symbol,
            current_price=request.current_price,
            historical_prices=request.historical_prices,
            volume=request.volume or 0.0,
            market_cap=request.market_cap or 0.0,
            earnings_per_share=request.earnings_per_share or 0.0,
            book_value_per_share=request.book_value_per_share or 0.0,
            dividend_yield=request.dividend_yield or 0.0,
            pe_ratio=request.pe_ratio or 0.0,
            pb_ratio=request.pb_ratio or 0.0,
            roe=request.roe or 0.0,
            debt_to_equity=request.debt_to_equity or 0.0,
            revenue_growth=request.revenue_growth or 0.0,
            free_cash_flow=request.free_cash_flow or 0.0,
            shares_outstanding=request.shares_outstanding or 0.0
        )
        
        start_time = datetime.now()
        
        if indicator_types:
            # Calculate specific indicators
            results = {}
            for indicator_type in indicator_types:
                try:
                    stock_indicator_type = StockIndicatorType(indicator_type)
                    result = stock_engine.calculate_indicator(stock_data, stock_indicator_type)
                    results[indicator_type] = IndicatorResponse(
                        indicator_type=result.indicator_type.value,
                        value=result.value,
                        confidence=result.confidence,
                        metadata=result.metadata,
                        timestamp=result.timestamp,
                        interpretation=result.interpretation,
                        risk_level=result.risk_level,
                        signal=result.signal,
                        time_horizon=result.time_horizon,
                        asset_symbols=result.asset_symbols
                    )
                except ValueError:
                    logger.warning(f"Unknown stock indicator type: {indicator_type}")
                    continue
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type}: {str(e)}")
                    continue
        else:
            # Calculate all indicators
            all_results = stock_engine.calculate_all_indicators(stock_data)
            results = {}
            for indicator_type, result in all_results.items():
                results[indicator_type.value] = IndicatorResponse(
                    indicator_type=result.indicator_type.value,
                    value=result.value,
                    confidence=result.confidence,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    interpretation=result.interpretation,
                    risk_level=result.risk_level,
                    signal=result.signal,
                    time_horizon=result.time_horizon,
                    asset_symbols=result.asset_symbols
                )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="stock",
            indicators=results,
            calculation_time=calculation_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating stock indicators: {str(e)}")
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="stock",
            indicators={},
            calculation_time=0.0,
            success=False,
            error_message=str(e)
        )

@router.post("/forex/calculate")
async def calculate_forex_indicators(
    request: ForexDataRequest,
    indicator_types: Optional[List[str]] = Query(None, description="Specific indicators to calculate")
):
    """Calculate forex-specific indicators"""
    try:
        if not forex_engine:
            raise HTTPException(status_code=500, detail="Forex indicator engine not available")
        
        # Convert to ForexAssetData (simplified conversion)
        forex_data = ForexAssetData(
            symbol=request.symbol,
            current_price=request.current_price,
            historical_prices=request.historical_prices,
            volume=request.volume or 0.0,
            base_currency=request.base_currency,
            quote_currency=request.quote_currency,
            interest_rate_base=request.interest_rate_base or 0.0,
            interest_rate_quote=request.interest_rate_quote or 0.0,
            inflation_rate_base=request.inflation_rate_base or 0.0,
            inflation_rate_quote=request.inflation_rate_quote or 0.0,
            gdp_growth_base=request.gdp_growth_base or 0.0,
            gdp_growth_quote=request.gdp_growth_quote or 0.0,
            trade_balance=request.trade_balance or 0.0,
            political_stability=request.political_stability or 0.0
        )
        
        start_time = datetime.now()
        
        if indicator_types:
            # Calculate specific indicators
            results = {}
            for indicator_type in indicator_types:
                try:
                    forex_indicator_type = ForexIndicatorType(indicator_type)
                    result = forex_engine.calculate_indicator(forex_data, forex_indicator_type)
                    results[indicator_type] = IndicatorResponse(
                        indicator_type=result.indicator_type.value,
                        value=result.value,
                        confidence=result.confidence,
                        metadata=result.metadata,
                        timestamp=result.timestamp,
                        interpretation=result.interpretation,
                        risk_level=result.risk_level,
                        signal=result.signal,
                        time_horizon=result.time_horizon,
                        asset_symbols=result.asset_symbols
                    )
                except ValueError:
                    logger.warning(f"Unknown forex indicator type: {indicator_type}")
                    continue
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type}: {str(e)}")
                    continue
        else:
            # Calculate all indicators
            all_results = forex_engine.calculate_all_indicators(forex_data)
            results = {}
            for indicator_type, result in all_results.items():
                results[indicator_type.value] = IndicatorResponse(
                    indicator_type=result.indicator_type.value,
                    value=result.value,
                    confidence=result.confidence,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    interpretation=result.interpretation,
                    risk_level=result.risk_level,
                    signal=result.signal,
                    time_horizon=result.time_horizon,
                    asset_symbols=result.asset_symbols
                )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="forex",
            indicators=results,
            calculation_time=calculation_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating forex indicators: {str(e)}")
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="forex",
            indicators={},
            calculation_time=0.0,
            success=False,
            error_message=str(e)
        )

@router.post("/futures/calculate")
async def calculate_futures_indicators(
    request: FuturesDataRequest,
    indicator_types: Optional[List[str]] = Query(None, description="Specific indicators to calculate")
):
    """Calculate futures-specific indicators"""
    try:
        if not futures_engine:
            raise HTTPException(status_code=500, detail="Futures indicator engine not available")
        
        # Convert to FuturesAssetData (simplified conversion)
        futures_data = FuturesAssetData(
            symbol=request.symbol,
            current_price=request.current_price,
            historical_prices=request.historical_prices,
            volume=request.volume or 0.0,
            underlying_asset=request.underlying_asset,
            expiration_date=request.expiration_date,
            contract_size=request.contract_size,
            spot_price=request.spot_price,
            risk_free_rate=request.risk_free_rate or 0.02,
            dividend_yield=request.dividend_yield or 0.0,
            storage_cost=request.storage_cost or 0.0,
            convenience_yield=request.convenience_yield or 0.0,
            open_interest=request.open_interest or 0.0,
            basis=request.basis or 0.0
        )
        
        start_time = datetime.now()
        
        if indicator_types:
            # Calculate specific indicators
            results = {}
            for indicator_type in indicator_types:
                try:
                    futures_indicator_type = FuturesIndicatorType(indicator_type)
                    result = futures_engine.calculate_indicator(futures_data, futures_indicator_type)
                    results[indicator_type] = IndicatorResponse(
                        indicator_type=result.indicator_type.value,
                        value=result.value,
                        confidence=result.confidence,
                        metadata=result.metadata,
                        timestamp=result.timestamp,
                        interpretation=result.interpretation,
                        risk_level=result.risk_level,
                        signal=result.signal,
                        time_horizon=result.time_horizon,
                        asset_symbols=result.asset_symbols
                    )
                except ValueError:
                    logger.warning(f"Unknown futures indicator type: {indicator_type}")
                    continue
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type}: {str(e)}")
                    continue
        else:
            # Calculate all indicators
            all_results = futures_engine.calculate_all_indicators(futures_data)
            results = {}
            for indicator_type, result in all_results.items():
                results[indicator_type.value] = IndicatorResponse(
                    indicator_type=result.indicator_type.value,
                    value=result.value,
                    confidence=result.confidence,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    interpretation=result.interpretation,
                    risk_level=result.risk_level,
                    signal=result.signal,
                    time_horizon=result.time_horizon,
                    asset_symbols=result.asset_symbols
                )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="futures",
            indicators=results,
            calculation_time=calculation_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating futures indicators: {str(e)}")
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="futures",
            indicators={},
            calculation_time=0.0,
            success=False,
            error_message=str(e)
        )

@router.post("/index/calculate")
async def calculate_index_indicators(
    request: IndexDataRequest,
    indicator_types: Optional[List[str]] = Query(None, description="Specific indicators to calculate")
):
    """Calculate index-specific indicators"""
    try:
        if not index_engine:
            raise HTTPException(status_code=500, detail="Index indicator engine not available")
        
        # Convert to IndexAssetData (simplified conversion)
        index_data = IndexAssetData(
            symbol=request.symbol,
            current_price=request.current_price,
            historical_prices=request.historical_prices,
            volume=request.volume or 0.0,
            constituent_symbols=request.constituent_symbols,
            constituent_weights=request.constituent_weights,
            dividend_yield=request.dividend_yield or 0.0,
            pe_ratio=request.pe_ratio or 0.0,
            sector_weights=request.sector_weights or {},
            market_cap_weighted=request.market_cap_weighted or True,
            rebalance_frequency=request.rebalance_frequency or "quarterly",
            expense_ratio=request.expense_ratio or 0.0
        )
        
        start_time = datetime.now()
        
        if indicator_types:
            # Calculate specific indicators
            results = {}
            for indicator_type in indicator_types:
                try:
                    index_indicator_type = IndexIndicatorType(indicator_type)
                    result = index_engine.calculate_indicator(index_data, index_indicator_type)
                    results[indicator_type] = IndicatorResponse(
                        indicator_type=result.indicator_type.value,
                        value=result.value,
                        confidence=result.confidence,
                        metadata=result.metadata,
                        timestamp=result.timestamp,
                        interpretation=result.interpretation,
                        risk_level=result.risk_level,
                        signal=result.signal,
                        time_horizon=result.time_horizon,
                        asset_symbols=result.asset_symbols
                    )
                except ValueError:
                    logger.warning(f"Unknown index indicator type: {indicator_type}")
                    continue
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type}: {str(e)}")
                    continue
        else:
            # Calculate all indicators
            all_results = index_engine.calculate_all_indicators(index_data)
            results = {}
            for indicator_type, result in all_results.items():
                results[indicator_type.value] = IndicatorResponse(
                    indicator_type=result.indicator_type.value,
                    value=result.value,
                    confidence=result.confidence,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    interpretation=result.interpretation,
                    risk_level=result.risk_level,
                    signal=result.signal,
                    time_horizon=result.time_horizon,
                    asset_symbols=result.asset_symbols
                )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="index",
            indicators=results,
            calculation_time=calculation_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating index indicators: {str(e)}")
        return BatchIndicatorResponse(
            asset_symbol=request.symbol,
            asset_type="index",
            indicators={},
            calculation_time=0.0,
            success=False,
            error_message=str(e)
        )

@router.post("/cross-asset/calculate")
async def calculate_cross_asset_indicators(
    request: IndicatorRequest,
    indicator_types: Optional[List[str]] = Query(None, description="Specific indicators to calculate")
):
    """Calculate cross-asset indicators"""
    try:
        if not cross_asset_engine:
            raise HTTPException(status_code=500, detail="Cross-asset indicator engine not available")
        
        asset_data = convert_to_asset_data(request.asset_data)
        additional_assets = [convert_to_asset_data(asset) for asset in request.additional_assets] if request.additional_assets else None
        
        start_time = datetime.now()
        
        if indicator_types:
            # Calculate specific indicators
            results = {}
            for indicator_type in indicator_types:
                try:
                    cross_asset_indicator_type = CrossAssetIndicatorType(indicator_type)
                    
                    # Handle different indicator types
                    if cross_asset_indicator_type == CrossAssetIndicatorType.MARKOWITZ_MPT and additional_assets:
                        result = cross_asset_engine.portfolio.markowitz_optimization([asset_data] + additional_assets)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.MONTE_CARLO:
                        result = cross_asset_engine.portfolio.monte_carlo_simulation(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.ARIMA:
                        result = cross_asset_engine.time_series.arima_forecast(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.GARCH:
                        result = cross_asset_engine.time_series.garch_volatility(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.RSI:
                        result = cross_asset_engine.technical.rsi(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.MACD:
                        result = cross_asset_engine.technical.macd(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.ICHIMOKU:
                        result = cross_asset_engine.technical.ichimoku_cloud(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.LSTM:
                        result = cross_asset_engine.ml_models.lstm_prediction(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.XGBOOST:
                        result = cross_asset_engine.ml_models.xgboost_prediction(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.BOLLINGER_BANDS:
                        result = cross_asset_engine._calculate_bollinger_bands(asset_data)
                    elif cross_asset_indicator_type == CrossAssetIndicatorType.STOCHASTIC:
                        result = cross_asset_engine._calculate_stochastic(asset_data)
                    else:
                        logger.warning(f"Unsupported cross-asset indicator: {indicator_type}")
                        continue
                    
                    results[indicator_type] = IndicatorResponse(
                        indicator_type=result.indicator_type.value,
                        value=result.value,
                        confidence=result.confidence,
                        metadata=result.metadata,
                        timestamp=result.timestamp,
                        interpretation=result.interpretation,
                        risk_level=result.risk_level,
                        signal=result.signal,
                        time_horizon=result.time_horizon,
                        asset_symbols=result.asset_symbols
                    )
                except ValueError:
                    logger.warning(f"Unknown cross-asset indicator type: {indicator_type}")
                    continue
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type}: {str(e)}")
                    continue
        else:
            # Calculate all indicators
            all_results = cross_asset_engine.calculate_all_indicators(asset_data, additional_assets)
            results = {}
            for indicator_type, result in all_results.items():
                results[indicator_type.value] = IndicatorResponse(
                    indicator_type=result.indicator_type.value,
                    value=result.value,
                    confidence=result.confidence,
                    metadata=result.metadata,
                    timestamp=result.timestamp,
                    interpretation=result.interpretation,
                    risk_level=result.risk_level,
                    signal=result.signal,
                    time_horizon=result.time_horizon,
                    asset_symbols=result.asset_symbols
                )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        return BatchIndicatorResponse(
            asset_symbol=request.asset_data.symbol,
            asset_type="cross_asset",
            indicators=results,
            calculation_time=calculation_time,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error calculating cross-asset indicators: {str(e)}")
        return BatchIndicatorResponse(
            asset_symbol=request.asset_data.symbol,
            asset_type="cross_asset",
            indicators={},
            calculation_time=0.0,
            success=False,
            error_message=str(e)
        )

@router.post("/batch/calculate")
async def calculate_batch_indicators(
    requests: List[IndicatorRequest],
    background_tasks: BackgroundTasks
):
    """Calculate indicators for multiple assets in batch"""
    try:
        results = []
        
        for request in requests:
            asset_type = request.asset_data.asset_type.lower()
            
            if asset_type == "crypto" and crypto_engine:
                # Convert to crypto request format
                crypto_request = CryptoDataRequest(**request.asset_data.dict())
                result = await calculate_crypto_indicators(crypto_request, request.indicator_types)
            elif asset_type == "stock" and stock_engine:
                # Convert to stock request format
                stock_request = StockDataRequest(**request.asset_data.dict())
                result = await calculate_stock_indicators(stock_request, request.indicator_types)
            elif asset_type == "forex" and forex_engine:
                # Convert to forex request format
                forex_request = ForexDataRequest(**request.asset_data.dict())
                result = await calculate_forex_indicators(forex_request, request.indicator_types)
            elif asset_type == "futures" and futures_engine:
                # Convert to futures request format
                futures_request = FuturesDataRequest(**request.asset_data.dict())
                result = await calculate_futures_indicators(futures_request, request.indicator_types)
            elif asset_type == "index" and index_engine:
                # Convert to index request format
                index_request = IndexDataRequest(**request.asset_data.dict())
                result = await calculate_index_indicators(index_request, request.indicator_types)
            else:
                # Use cross-asset engine as fallback
                result = await calculate_cross_asset_indicators(request, request.indicator_types)
            
            results.append(result)
        
        return {
            "batch_results": results,
            "total_assets": len(requests),
            "successful_calculations": sum(1 for r in results if r.success),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in batch calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics for indicator calculations"""
    try:
        # This would typically come from a monitoring system
        # For now, return mock metrics
        return {
            "total_calculations": 1000,
            "average_calculation_time": 0.25,
            "success_rate": 0.98,
            "engine_status": {
                "crypto": "healthy" if crypto_engine else "unavailable",
                "stock": "healthy" if stock_engine else "unavailable",
                "forex": "healthy" if forex_engine else "unavailable",
                "futures": "healthy" if futures_engine else "unavailable",
                "index": "healthy" if index_engine else "unavailable",
                "cross_asset": "healthy" if cross_asset_engine else "unavailable"
            },
            "most_used_indicators": [
                "rsi", "macd", "bollinger_bands", "arima", "lstm"
            ],
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))