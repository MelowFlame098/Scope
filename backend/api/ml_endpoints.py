"""Machine Learning API Endpoints

Provides REST API endpoints for all AI/ML capabilities in Phase 5:
- Technical analysis indicators and signals
- Statistical models and forecasting
- Machine learning pipelines
- Model orchestration and management
- Reinforcement learning agents
- Model performance monitoring
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

# Import our ML services
try:
    from ..technical_analysis import technical_analysis_service, IndicatorType, SignalType
    from ..statistical_models import statistical_models_service, ModelType as StatModelType
    from ..ml_pipeline import ml_pipeline_service, ModelType as MLModelType, ModelConfig
    from ..model_orchestrator import model_orchestrator, ModelCategory, PredictionType, ModelStatus
    from ..reinforcement_learning import rl_service, AgentType, ActionType
except ImportError:
    # Fallback for development
    technical_analysis_service = None
    statistical_models_service = None
    ml_pipeline_service = None
    model_orchestrator = None
    rl_service = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

# Pydantic models for request/response
class MarketDataRequest(BaseModel):
    """Market data for analysis"""
    symbol: str
    data: List[Dict[str, Any]]
    timeframe: str = "1d"
    
class TechnicalAnalysisRequest(BaseModel):
    """Technical analysis request"""
    symbol: str
    indicator: str
    data: List[Dict[str, Any]]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class TechnicalAnalysisResponse(BaseModel):
    """Technical analysis response"""
    symbol: str
    indicator: str
    values: List[float]
    signals: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
class StatisticalModelRequest(BaseModel):
    """Statistical model request"""
    model_type: str
    target_column: str
    data: List[Dict[str, Any]]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    forecast_periods: int = 30
    
class StatisticalModelResponse(BaseModel):
    """Statistical model response"""
    model_type: str
    predictions: List[float]
    confidence_intervals: List[List[float]]
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    
class MLModelRequest(BaseModel):
    """ML model training request"""
    model_type: str
    target_column: str
    feature_columns: List[str]
    data: List[Dict[str, Any]]
    config: Dict[str, Any] = Field(default_factory=dict)
    
class MLModelResponse(BaseModel):
    """ML model response"""
    model_id: str
    model_type: str
    predictions: List[float]
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]
    
class ModelPredictionRequest(BaseModel):
    """Model prediction request"""
    model_id: str
    data: List[Dict[str, Any]]
    
class ModelPredictionResponse(BaseModel):
    """Model prediction response"""
    model_id: str
    prediction_type: str
    value: Union[float, int, str, Dict[str, Any]]
    confidence: float
    timestamp: datetime
    features_used: List[str]
    metadata: Dict[str, Any]
    
class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction response"""
    ensemble_id: str
    individual_predictions: List[ModelPredictionResponse]
    final_prediction: ModelPredictionResponse
    weights: Dict[str, float]
    consensus_score: float
    disagreement_score: float
    
class RLAgentRequest(BaseModel):
    """RL agent creation request"""
    agent_id: str
    agent_type: str = "dqn"
    state_size: int = 20
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class RLTrainingRequest(BaseModel):
    """RL training request"""
    agent_id: str
    training_data: List[Dict[str, Any]]
    episodes: int = 1000
    validation_split: float = 0.2
    
class RLSignalRequest(BaseModel):
    """RL trading signal request"""
    agent_id: str
    current_data: List[Dict[str, Any]]
    
class RLSignalResponse(BaseModel):
    """RL trading signal response"""
    action_type: str
    symbol: str
    quantity: float
    price: Optional[float]
    confidence: float
    metadata: Dict[str, Any]

# Helper functions
def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert request data to pandas DataFrame"""
    df = pd.DataFrame(data)
    
    # Convert timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def handle_service_error(service_name: str, error: Exception):
    """Handle service errors consistently"""
    logger.error(f"Error in {service_name}: {str(error)}")
    raise HTTPException(
        status_code=500,
        detail=f"{service_name} error: {str(error)}"
    )

# Technical Analysis Endpoints
@router.post("/technical-analysis/indicator", response_model=TechnicalAnalysisResponse)
async def calculate_technical_indicator(
    request: TechnicalAnalysisRequest
) -> TechnicalAnalysisResponse:
    """Calculate technical analysis indicator"""
    if not technical_analysis_service:
        raise HTTPException(status_code=503, detail="Technical analysis service not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        # Map indicator name to enum
        indicator_map = {
            'sma': IndicatorType.SMA,
            'ema': IndicatorType.EMA,
            'rsi': IndicatorType.RSI,
            'macd': IndicatorType.MACD,
            'bollinger_bands': IndicatorType.BOLLINGER_BANDS,
            'stochastic': IndicatorType.STOCHASTIC,
            'williams_r': IndicatorType.WILLIAMS_R,
            'cci': IndicatorType.CCI,
            'atr': IndicatorType.ATR,
            'adx': IndicatorType.ADX
        }
        
        indicator_type = indicator_map.get(request.indicator.lower())
        if not indicator_type:
            raise HTTPException(status_code=400, detail=f"Unknown indicator: {request.indicator}")
        
        result = await technical_analysis_service.calculate_indicator(
            df, indicator_type, **request.parameters
        )
        
        # Convert signals to dict format
        signals = []
        for signal in result.signals:
            signals.append({
                'timestamp': signal.timestamp.isoformat(),
                'signal_type': signal.signal_type.value,
                'strength': signal.strength,
                'price': signal.price,
                'description': signal.description,
                'metadata': signal.metadata
            })
        
        # Convert patterns to dict format
        patterns = []
        for pattern in result.patterns:
            patterns.append({
                'pattern_type': pattern.pattern_type.value,
                'confidence': pattern.confidence,
                'start_time': pattern.start_time.isoformat(),
                'end_time': pattern.end_time.isoformat(),
                'key_points': pattern.key_points,
                'description': pattern.description,
                'metadata': pattern.metadata
            })
        
        return TechnicalAnalysisResponse(
            symbol=request.symbol,
            indicator=request.indicator,
            values=result.values.tolist() if result.values is not None else [],
            signals=signals,
            patterns=patterns,
            metadata=result.metadata
        )
        
    except Exception as e:
        handle_service_error("Technical Analysis", e)

@router.post("/technical-analysis/signals")
async def get_trading_signals(
    request: MarketDataRequest
) -> Dict[str, Any]:
    """Get comprehensive trading signals"""
    if not technical_analysis_service:
        raise HTTPException(status_code=503, detail="Technical analysis service not available")
    
    try:
        df = convert_to_dataframe(request.data)
        signals = await technical_analysis_service.get_trading_signals(df)
        
        # Convert to serializable format
        result = []
        for signal in signals:
            result.append({
                'timestamp': signal.timestamp.isoformat(),
                'signal_type': signal.signal_type.value,
                'strength': signal.strength,
                'price': signal.price,
                'description': signal.description,
                'metadata': signal.metadata
            })
        
        return {
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'signals': result,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        handle_service_error("Technical Analysis Signals", e)

# Statistical Models Endpoints
@router.post("/statistical/model", response_model=StatisticalModelResponse)
async def fit_statistical_model(
    request: StatisticalModelRequest
) -> StatisticalModelResponse:
    """Fit statistical model and generate predictions"""
    if not statistical_models_service:
        raise HTTPException(status_code=503, detail="Statistical models service not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        # Map model type
        model_map = {
            'arima': StatModelType.ARIMA,
            'garch': StatModelType.GARCH,
            'var': StatModelType.VAR,
            'linear_regression': StatModelType.LINEAR_REGRESSION,
            'polynomial_regression': StatModelType.POLYNOMIAL_REGRESSION,
            'ridge_regression': StatModelType.RIDGE_REGRESSION,
            'lasso_regression': StatModelType.LASSO_REGRESSION,
            'monte_carlo': StatModelType.MONTE_CARLO
        }
        
        model_type = model_map.get(request.model_type.lower())
        if not model_type:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
        
        result = await statistical_models_service.fit_model(
            df, model_type, request.target_column, **request.parameters
        )
        
        # Generate forecasts if requested
        predictions = []
        confidence_intervals = []
        
        if result.predictions is not None:
            predictions = result.predictions.tolist()
        
        if hasattr(result, 'confidence_intervals') and result.confidence_intervals is not None:
            confidence_intervals = result.confidence_intervals.tolist()
        
        return StatisticalModelResponse(
            model_type=request.model_type,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            metrics=result.metrics,
            diagnostics=result.diagnostics
        )
        
    except Exception as e:
        handle_service_error("Statistical Models", e)

@router.post("/statistical/forecast")
async def generate_forecast(
    request: StatisticalModelRequest
) -> Dict[str, Any]:
    """Generate statistical forecast"""
    if not statistical_models_service:
        raise HTTPException(status_code=503, detail="Statistical models service not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        # Use ARIMA as default for forecasting
        result = await statistical_models_service.fit_model(
            df, StatModelType.ARIMA, request.target_column, **request.parameters
        )
        
        forecast = await statistical_models_service.generate_forecast(
            result.model, request.forecast_periods
        )
        
        return {
            'forecast': forecast.tolist() if forecast is not None else [],
            'model_metrics': result.metrics,
            'forecast_periods': request.forecast_periods,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        handle_service_error("Statistical Forecast", e)

# Machine Learning Endpoints
@router.post("/ml/train", response_model=MLModelResponse)
async def train_ml_model(
    request: MLModelRequest,
    background_tasks: BackgroundTasks
) -> MLModelResponse:
    """Train machine learning model"""
    if not ml_pipeline_service:
        raise HTTPException(status_code=503, detail="ML pipeline service not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        # Map model type
        model_map = {
            'linear_regression': MLModelType.LINEAR_REGRESSION,
            'random_forest': MLModelType.RANDOM_FOREST,
            'gradient_boosting': MLModelType.GRADIENT_BOOSTING,
            'svm': MLModelType.SVM,
            'neural_network': MLModelType.NEURAL_NETWORK,
            'xgboost': MLModelType.XGBOOST,
            'lightgbm': MLModelType.LIGHTGBM
        }
        
        model_type = model_map.get(request.model_type.lower())
        if not model_type:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
        
        # Create model config
        config = ModelConfig(
            model_type=model_type,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            **request.config
        )
        
        # Train model
        result = await ml_pipeline_service.train_model(df, config)
        
        return MLModelResponse(
            model_id=result.model_id,
            model_type=request.model_type,
            predictions=result.predictions.tolist() if result.predictions is not None else [],
            feature_importance=result.feature_importance,
            metrics=result.metrics
        )
        
    except Exception as e:
        handle_service_error("ML Training", e)

@router.post("/ml/predict")
async def predict_with_ml_model(
    request: ModelPredictionRequest
) -> Dict[str, Any]:
    """Make predictions with trained ML model"""
    if not ml_pipeline_service:
        raise HTTPException(status_code=503, detail="ML pipeline service not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        predictions = await ml_pipeline_service.predict(
            request.model_id, df
        )
        
        return {
            'model_id': request.model_id,
            'predictions': predictions.tolist() if predictions is not None else [],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        handle_service_error("ML Prediction", e)

# Model Orchestration Endpoints
@router.get("/models")
async def list_models(
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    prediction_type: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """List available models"""
    if not model_orchestrator:
        raise HTTPException(status_code=503, detail="Model orchestrator not available")
    
    try:
        # Convert string parameters to enums
        category_enum = None
        if category:
            category_enum = ModelCategory(category.lower())
        
        status_enum = None
        if status:
            status_enum = ModelStatus(status.lower())
        
        prediction_type_enum = None
        if prediction_type:
            prediction_type_enum = PredictionType(prediction_type.lower())
        
        models = model_orchestrator.list_models(
            category=category_enum,
            status=status_enum,
            prediction_type=prediction_type_enum
        )
        
        # Convert to serializable format
        result = []
        for model in models:
            result.append({
                'model_id': model.model_id,
                'name': model.name,
                'category': model.category.value,
                'prediction_type': model.prediction_type.value,
                'version': model.version,
                'status': model.status.value,
                'description': model.description,
                'created_at': model.created_at.isoformat(),
                'updated_at': model.updated_at.isoformat(),
                'performance_metrics': model.performance_metrics
            })
        
        return {
            'models': result,
            'total_count': len(result)
        }
        
    except Exception as e:
        handle_service_error("Model Orchestrator", e)

@router.post("/models/predict", response_model=ModelPredictionResponse)
async def predict_with_orchestrator(
    request: ModelPredictionRequest
) -> ModelPredictionResponse:
    """Generate prediction using model orchestrator"""
    if not model_orchestrator:
        raise HTTPException(status_code=503, detail="Model orchestrator not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        prediction = await model_orchestrator.predict(
            request.model_id, df
        )
        
        return ModelPredictionResponse(
            model_id=prediction.model_id,
            prediction_type=prediction.prediction_type.value,
            value=prediction.value,
            confidence=prediction.confidence,
            timestamp=prediction.timestamp,
            features_used=prediction.features_used,
            metadata=prediction.metadata
        )
        
    except Exception as e:
        handle_service_error("Model Orchestrator Prediction", e)

@router.post("/models/ensemble/predict", response_model=EnsemblePredictionResponse)
async def predict_with_ensemble(
    ensemble_id: str,
    request: ModelPredictionRequest
) -> EnsemblePredictionResponse:
    """Generate ensemble prediction"""
    if not model_orchestrator:
        raise HTTPException(status_code=503, detail="Model orchestrator not available")
    
    try:
        df = convert_to_dataframe(request.data)
        
        ensemble_prediction = await model_orchestrator.predict_ensemble(
            ensemble_id, df
        )
        
        # Convert individual predictions
        individual_preds = []
        for pred in ensemble_prediction.individual_predictions:
            individual_preds.append(ModelPredictionResponse(
                model_id=pred.model_id,
                prediction_type=pred.prediction_type.value,
                value=pred.value,
                confidence=pred.confidence,
                timestamp=pred.timestamp,
                features_used=pred.features_used,
                metadata=pred.metadata
            ))
        
        # Convert final prediction
        final_pred = ModelPredictionResponse(
            model_id=ensemble_prediction.final_prediction.model_id,
            prediction_type=ensemble_prediction.final_prediction.prediction_type.value,
            value=ensemble_prediction.final_prediction.value,
            confidence=ensemble_prediction.final_prediction.confidence,
            timestamp=ensemble_prediction.final_prediction.timestamp,
            features_used=ensemble_prediction.final_prediction.features_used,
            metadata=ensemble_prediction.final_prediction.metadata
        )
        
        return EnsemblePredictionResponse(
            ensemble_id=ensemble_prediction.ensemble_id,
            individual_predictions=individual_preds,
            final_prediction=final_pred,
            weights=ensemble_prediction.weights,
            consensus_score=ensemble_prediction.consensus_score,
            disagreement_score=ensemble_prediction.disagreement_score
        )
        
    except Exception as e:
        handle_service_error("Ensemble Prediction", e)

@router.get("/models/{model_id}/performance")
async def get_model_performance(
    model_id: str
) -> Dict[str, Any]:
    """Get model performance history"""
    if not model_orchestrator:
        raise HTTPException(status_code=503, detail="Model orchestrator not available")
    
    try:
        performance_history = model_orchestrator.get_model_performance_history(model_id)
        
        result = []
        for perf in performance_history:
            result.append({
                'evaluation_date': perf.evaluation_date.isoformat(),
                'metrics': perf.metrics,
                'prediction_accuracy': perf.prediction_accuracy,
                'latency_ms': perf.latency_ms,
                'throughput_rps': perf.throughput_rps,
                'error_rate': perf.error_rate,
                'drift_score': perf.drift_score
            })
        
        return {
            'model_id': model_id,
            'performance_history': result
        }
        
    except Exception as e:
        handle_service_error("Model Performance", e)

# Reinforcement Learning Endpoints
@router.post("/rl/agents")
async def create_rl_agent(
    request: RLAgentRequest
) -> Dict[str, Any]:
    """Create new RL agent"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
    
    try:
        agent_type_map = {
            'dqn': AgentType.DQN,
            'ddpg': AgentType.DDPG,
            'ppo': AgentType.PPO,
            'a3c': AgentType.A3C
        }
        
        agent_type = agent_type_map.get(request.agent_type.lower(), AgentType.DQN)
        
        agent_id = await rl_service.create_agent(
            request.agent_id,
            agent_type,
            request.state_size,
            **request.parameters
        )
        
        return {
            'agent_id': agent_id,
            'agent_type': request.agent_type,
            'state_size': request.state_size,
            'created_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        handle_service_error("RL Agent Creation", e)

@router.post("/rl/agents/{agent_id}/train")
async def train_rl_agent(
    agent_id: str,
    request: RLTrainingRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Train RL agent"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
    
    try:
        df = convert_to_dataframe(request.training_data)
        
        # Start training in background
        async def train_agent():
            await rl_service.train_agent(
                agent_id,
                df,
                request.episodes,
                request.validation_split
            )
        
        background_tasks.add_task(train_agent)
        
        return {
            'agent_id': agent_id,
            'training_started': True,
            'episodes': request.episodes,
            'validation_split': request.validation_split,
            'started_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        handle_service_error("RL Training", e)

@router.post("/rl/agents/{agent_id}/signal", response_model=RLSignalResponse)
async def get_rl_trading_signal(
    agent_id: str,
    request: RLSignalRequest
) -> RLSignalResponse:
    """Get trading signal from RL agent"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
    
    try:
        df = convert_to_dataframe(request.current_data)
        
        signal = await rl_service.get_trading_signal(agent_id, df)
        
        return RLSignalResponse(
            action_type=signal.action_type.value,
            symbol=signal.symbol,
            quantity=signal.quantity,
            price=signal.price,
            confidence=signal.confidence,
            metadata=signal.metadata
        )
        
    except Exception as e:
        handle_service_error("RL Trading Signal", e)

@router.get("/rl/agents")
async def list_rl_agents() -> Dict[str, Any]:
    """List all RL agents"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
    
    try:
        agents = rl_service.list_agents()
        
        result = []
        for agent_id in agents:
            info = rl_service.get_agent_info(agent_id)
            result.append({
                'agent_id': agent_id,
                'agent_type': info['agent_type'].value,
                'created_at': info['created_at'].isoformat(),
                'state_size': info['state_size'],
                'training_episodes': info['training_episodes'],
                'performance_history_count': len(info['performance_history'])
            })
        
        return {
            'agents': result,
            'total_count': len(result)
        }
        
    except Exception as e:
        handle_service_error("RL Agents List", e)

@router.get("/rl/agents/{agent_id}/performance")
async def get_rl_agent_performance(
    agent_id: str
) -> Dict[str, Any]:
    """Get RL agent performance"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
    
    try:
        info = rl_service.get_agent_info(agent_id)
        
        performance_history = []
        for perf in info['performance_history']:
            performance_history.append({
                'evaluation_date': perf.evaluation_date.isoformat(),
                'total_return': perf.total_return,
                'annualized_return': perf.annualized_return,
                'volatility': perf.volatility,
                'sharpe_ratio': perf.sharpe_ratio,
                'max_drawdown': perf.max_drawdown,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'num_trades': perf.num_trades,
                'risk_adjusted_return': perf.risk_adjusted_return
            })
        
        return {
            'agent_id': agent_id,
            'agent_info': {
                'agent_type': info['agent_type'].value,
                'created_at': info['created_at'].isoformat(),
                'state_size': info['state_size'],
                'training_episodes': info['training_episodes']
            },
            'performance_history': performance_history
        }
        
    except Exception as e:
        handle_service_error("RL Agent Performance", e)

# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check health of all ML services"""
    services_status = {
        'technical_analysis': technical_analysis_service is not None,
        'statistical_models': statistical_models_service is not None,
        'ml_pipeline': ml_pipeline_service is not None,
        'model_orchestrator': model_orchestrator is not None,
        'reinforcement_learning': rl_service is not None
    }
    
    all_healthy = all(services_status.values())
    
    return {
        'status': 'healthy' if all_healthy else 'degraded',
        'services': services_status,
        'timestamp': datetime.now().isoformat()
    }