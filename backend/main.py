from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
import json
from datetime import datetime
import logging

from database import get_db, init_db
from db_models import *
from schemas import *
from auth import get_current_user, create_access_token, verify_password, get_password_hash
from market_data import MarketDataService
from news_service import NewsService
from ai_service import AIService
from portfolio_service import PortfolioService
from forum_service import ForumService
from trading_service import TradingService
from notification_service import NotificationService
from paper_trading_service import PaperTradingService
from websocket_manager import WebSocketManager

# Phase 3: Enhanced services
from enhanced_market_data import enhanced_market_data_service
from real_time_streaming import real_time_streaming_service
from data_normalization import data_normalization_service

# Phase 4: News & Sentiment Engine
from sentiment_engine import get_sentiment_engine
from enhanced_news_service import get_enhanced_news_service
from content_filter import get_content_filter_service

# Phase 5: AI/ML Model Ecosystem
from technical_analysis import TechnicalAnalysisService
from statistical_models import StatisticalModelsService
from ml_pipeline import MLPipelineService
from model_orchestrator import ModelOrchestrator
from reinforcement_learning import ReinforcementLearningService

# Phase 6: LLM Explanation Engine
from llm_service import LLMService, LLMProvider, ExplanationRequest, ExplanationResponse
from prompt_engine import PromptEngine, PromptType, ExplanationComplexity
from conversation_manager import ConversationManager, ConversationRequest, ConversationResponse
from explanation_generator import ExplanationGenerator, FinancialExplanationRequest, FinancialExplanationResponse
from financial_context import FinancialContextEngine, ContextType

# Conditional LangChain import
try:
    from langchain_integration import LangChainIntegration, LangChainRequest, LangChainResponse, ChainType, AgentType
    LANGCHAIN_INTEGRATION_AVAILABLE = True
except ImportError:
    LANGCHAIN_INTEGRATION_AVAILABLE = False
    logger.warning("LangChain integration not available")

# Phase 7: Advanced Features
from portfolio_manager import PortfolioManager, PortfolioRequest, PortfolioResponse, TransactionRequest, RebalanceRequest
from portfolio_analytics import PortfolioAnalytics
from risk_engine import RiskEngine, RiskRequest, RiskLimitRequest
from trading_engine import TradingEngine, TradingRequest, OrderRequest, OrderResponse
from community_forum import CommunityForum, PostRequest, CommentRequest, PostResponse
from notification_system import NotificationSystem, NotificationRequest
from advanced_charting import AdvancedCharting, ChartRequest, IndicatorRequest, PatternScanRequest

# Phase 8: Enterprise-Grade Features
from ai_trading_strategies import AITradingStrategiesService, StrategyRequest, StrategyResponse, SignalRequest
from social_trading import SocialTradingService, router as social_trading_router
from institutional import InstitutionalService, router as institutional_router
from regulatory import RegulatoryService, router as regulatory_router
from mobile_api import MobileAPIService, router as mobile_api_router
from enterprise_security import EnterpriseSecurityService, router as enterprise_security_router
from compliance_engine import ComplianceEngine, router as compliance_engine_router
from institutional_analytics import InstitutionalAnalytics, router as institutional_analytics_router

# Phase 9: AI-First Platform
try:
    from ai_core import (
        AICore,
        AutonomousTradingSystem,
        ConversationalAI,
        PersonalizationEngine,
        MarketForecaster,
        RiskPredictor,
        TrendAnalyzer,
        ScenarioAnalyzer,
        EconomicForecaster,
        SentimentPredictor
    )
    AI_CORE_AVAILABLE = True
    logging.info("AI Core modules imported successfully")
except ImportError as e:
    AI_CORE_AVAILABLE = False
    logging.warning(f"AI Core modules not available: {e}")
    AICore = None

# Phase 10: Decentralized Finance Integration
try:
    from defi_core import (
        DeFiCore,
        ProtocolIntegrator,
        YieldOptimizer,
        LiquidityManager,
        CrossChainBridge,
        GasOptimizer,
        NFTAnalyzer,
        BlockchainAnalytics,
        DecentralizedIdentity
    )
    DEFI_CORE_AVAILABLE = True
    logging.info("DeFi Core modules imported successfully")
except ImportError as e:
    DEFI_CORE_AVAILABLE = False
    logging.warning(f"DeFi Core modules not available: {e}")
    DeFiCore = None

# API Router
from api.main import api_router
from api.models import router as models_router
from api.paper_trading import router as paper_trading_router

# Initialize services
market_data_service = MarketDataService()
news_service = NewsService()
ai_service = AIService()
portfolio_service = PortfolioService()
forum_service = ForumService()
trading_service = TradingService()
notification_service = NotificationService()
paper_trading_service = PaperTradingService()

# Phase 5: AI/ML Services
technical_analysis_service = TechnicalAnalysisService()
statistical_models_service = StatisticalModelsService()
ml_pipeline_service = MLPipelineService()
model_orchestrator = ModelOrchestrator()
rl_service = ReinforcementLearningService()

# Phase 6: LLM Services
llm_service = LLMService()
prompt_engine = PromptEngine()
conversation_manager = ConversationManager()
explanation_generator = ExplanationGenerator()
financial_context_engine = FinancialContextEngine()
langchain_integration = LangChainIntegration() if LANGCHAIN_INTEGRATION_AVAILABLE else None

# Phase 7: Advanced Services
portfolio_manager = PortfolioManager()
portfolio_analytics = PortfolioAnalytics()
risk_engine = RiskEngine()
trading_engine = TradingEngine()
community_forum = CommunityForum()
notification_system = NotificationSystem()
advanced_charting = AdvancedCharting()

# Phase 8: Enterprise Services
ai_trading_strategies_service = AITradingStrategiesService()
social_trading_service = SocialTradingService()
institutional_service = InstitutionalService()
regulatory_service = RegulatoryService()
mobile_api_service = MobileAPIService()
enterprise_security_service = EnterpriseSecurityService()
compliance_engine = ComplianceEngine()
institutional_analytics = InstitutionalAnalytics()

# Phase 9: AI-First Platform Services
ai_core_engine = None
if AI_CORE_AVAILABLE and AICore:
    try:
        ai_core_config = {
            'autonomous_trading': {
                'enabled': True,
                'risk_tolerance': 0.05,
                'max_position_size': 0.1
            },
            'natural_language': {
                'enabled': True,
                'model_name': 'gpt-4',
                'max_tokens': 2048
            },
            'personalization': {
                'enabled': True,
                'learning_rate': 0.01,
                'adaptation_threshold': 0.8
            }
        }
        ai_core_engine = AICore(config=ai_core_config)
        logger.info("AI Core Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Core Engine: {e}")
        ai_core_engine = None
else:
    logger.warning("AI Core Engine not available - running in legacy mode")

# Phase 10: DeFi Platform Services
defi_core_engine = None
if DEFI_CORE_AVAILABLE and DeFiCore:
    try:
        defi_core_config = {
            'protocols': {
                'ethereum_rpc': 'https://mainnet.infura.io/v3/your-project-id',
                'polygon_rpc': 'https://polygon-rpc.com',
                'bsc_rpc': 'https://bsc-dataseed.binance.org',
                'supported_protocols': ['uniswap', 'compound', 'aave', 'curve']
            },
            'yield_optimization': {
                'min_apy': 5.0,
                'max_risk_level': 'medium',
                'rebalance_threshold': 0.1
            },
            'liquidity': {
                'min_liquidity_usd': 1000000,
                'impermanent_loss_threshold': 0.05,
                'auto_rebalance': True
            },
            'cross_chain': {
                'supported_bridges': ['polygon', 'arbitrum', 'optimism'],
                'max_slippage': 0.01,
                'security_checks': True
            },
            'gas_optimization': {
                'target_gas_price': 'standard',
                'max_gas_price_gwei': 100,
                'optimization_enabled': True
            },
            'nft_analysis': {
                'supported_marketplaces': ['opensea', 'looksrare', 'x2y2'],
                'rarity_calculation': True,
                'price_prediction': True
            },
            'blockchain_analytics': {
                'supported_chains': ['ethereum', 'polygon', 'bsc', 'arbitrum'],
                'transaction_monitoring': True,
                'wallet_analytics': True
            },
            'decentralized_identity': {
                'did_registry': {
                    'ethereum_registry': '0x0123456789abcdef0123456789abcdef01234567'
                },
                'authentication_settings': {
                    'challenge_expiry_minutes': 15,
                    'session_expiry_hours': 24,
                    'require_2fa': False
                },
                'privacy_settings': {
                    'default_privacy_level': 'pseudonymous',
                    'zero_knowledge_enabled': True
                }
            }
        }
        defi_core_engine = DeFiCore(config=defi_core_config)
        logger.info("DeFi Core Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DeFi Core Engine: {e}")
        defi_core_engine = None
else:
    logger.warning("DeFi Core Engine not available - running without DeFi features")

# Initialize database
init_db()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinScope API", 
    version="2.0.0",
    description="Comprehensive Financial Platform API"
)

# Initialize WebSocket manager
ws_manager = WebSocketManager()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include API router
app.include_router(api_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(paper_trading_router, prefix="/api")

# Include Phase 8 routers
app.include_router(social_trading_router)
app.include_router(institutional_router)
app.include_router(regulatory_router)
app.include_router(mobile_api_router)
app.include_router(enterprise_security_router)
app.include_router(compliance_engine_router)
app.include_router(institutional_analytics_router)

# Services
market_service = MarketDataService()
news_service = NewsService()
ai_service = AIService()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services and start background tasks"""
    logger.info("🚀 FinScope Backend Starting...")
    logger.info("📊 Phase 3: Market Data Integration Active")
    logger.info("📰 Phase 4: News & Sentiment Engine Active")
    logger.info("🤖 Phase 5: AI/ML Model Ecosystem Active")
    logger.info("🧠 Phase 6: LLM Explanation Engine Active")
    logger.info("🔄 Real-time streaming service initialized")
    logger.info("💾 Enhanced caching with Redis enabled")
    logger.info("🌐 Multi-source data normalization ready")
    logger.info("🧠 AI-powered sentiment analysis ready")
    logger.info("📈 Enhanced news aggregation active")
    logger.info("📊 Technical analysis indicators ready")
    logger.info("📈 Statistical models and forecasting active")
    logger.info("🤖 Machine learning pipeline initialized")
    logger.info("🎯 Reinforcement learning agents ready")
    logger.info("🎛️ Model orchestration system active")
    logger.info("💬 LLM explanation engine ready")
    logger.info("🔗 LangChain integration initialized")
    logger.info("🏢 Phase 8: Enterprise-Grade Features Active")
    logger.info("🤖 AI trading strategies service ready")
    logger.info("👥 Social trading platform initialized")
    logger.info("🏛️ Institutional tools and analytics active")
    logger.info("⚖️ Regulatory compliance engine ready")
    logger.info("📱 Mobile API service initialized")
    logger.info("🔒 Enterprise security features active")
    logger.info("📋 Compliance monitoring system ready")
    logger.info("📝 Conversation management active")
    logger.info("🎯 Financial context engine ready")
    
    # Phase 9: AI-First Platform Status
    if ai_core_engine:
        logger.info("🧠 Phase 9: AI-First Platform Active")
        logger.info("🤖 AI Core Engine initialized")
        status = ai_core_engine.get_system_status()
        for component, available in status['components'].items():
            status_icon = "✅" if available else "❌"
            logger.info(f"{status_icon} {component.replace('_', ' ').title()}: {'Available' if available else 'Unavailable'}")
        logger.info(f"🎯 Available AI Features: {len(status['available_features'])}")
    else:
        logger.info("⚠️ Phase 9: AI-First Platform - Running in Legacy Mode")
    
    # Phase 10: DeFi Platform Status
    if defi_core_engine:
        logger.info("🏦 Phase 10: Decentralized Finance Integration Active")
        logger.info("⛓️ DeFi Core Engine initialized")
        status = defi_core_engine.get_status()
        for component, available in status['components'].items():
            status_icon = "✅" if available else "❌"
            logger.info(f"{status_icon} {component.replace('_', ' ').title()}: {'Available' if available else 'Unavailable'}")
        logger.info(f"🎯 Available DeFi Features: {len(defi_core_engine.get_features())}")
    else:
        logger.info("⚠️ Phase 10: DeFi Platform - Running without DeFi features")
    
    # Start background tasks
    asyncio.create_task(ws_manager.start_background_tasks())
    asyncio.create_task(notification_service.start_background_tasks())
    asyncio.create_task(trading_service.start_background_tasks())
    
    logger.info("✅ FinScope Backend Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down FinScope Backend...")
    
    # Cleanup Phase 4 services
    try:
        sentiment_engine = await get_sentiment_engine()
        await sentiment_engine.cleanup()
        
        news_service = await get_enhanced_news_service()
        await news_service.cleanup()
        
        filter_service = await get_content_filter_service()
        await filter_service.cleanup()
    except Exception as e:
        logger.error(f"Error during Phase 4 cleanup: {e}")
    
    # Cleanup enhanced services
    await enhanced_market_data_service.cleanup()
    await real_time_streaming_service.cleanup()
    
    # Cleanup AI core
    if ai_core_engine:
        try:
            await ai_core_engine.shutdown()
            logger.info("🧠 AI Core Engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during AI core shutdown: {e}")
    
    # Cleanup DeFi core
    if defi_core_engine:
        try:
            await defi_core_engine.shutdown()
            logger.info("⛓️ DeFi Core Engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during DeFi core shutdown: {e}")
    
    await ws_manager.cleanup()
    logger.info("✅ FinScope Backend Shutdown Complete")

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Send welcome notification
    await notification_service.send_welcome_notification(db_user.id, db_user.email)
    
    return UserResponse(
        id=db_user.id,
        email=db_user.email,
        username=db_user.username,
        created_at=db_user.created_at
    )

@app.post("/auth/login", response_model=Token)
async def login(form_data: UserLogin, db: Session = Depends(get_db)):
    """Login user and return access token"""
    user = db.query(User).filter(User.email == form_data.email).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.email})
    return Token(access_token=access_token, token_type="bearer")

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        created_at=current_user.created_at
    )

# ============================================================================
# MARKET DATA ENDPOINTS (V2)
# ============================================================================

@app.get("/api/v2/market-data/price/{symbol}")
async def get_real_time_price(symbol: str):
    """Get real-time price for a specific symbol"""
    try:
        price_data = await enhanced_market_data_service.get_real_time_prices([symbol])
        if symbol in price_data:
            return {
                "symbol": symbol,
                "price": price_data[symbol].get('price'),
                "change": price_data[symbol].get('change_24h'),
                "change_percent": price_data[symbol].get('change_percentage_24h'),
                "volume": price_data[symbol].get('volume_24h'),
                "high_24h": price_data[symbol].get('high_24h'),
                "low_24h": price_data[symbol].get('low_24h'),
                "market_cap": price_data[symbol].get('market_cap'),
                "timestamp": price_data[symbol].get('timestamp')
            }
        else:
            raise HTTPException(status_code=404, detail=f"Price data for {symbol} not found")
    except Exception as e:
        logger.error(f"Error fetching real-time price for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch real-time price")

@app.post("/api/v2/market-data/prices")
async def get_multiple_real_time_prices(request: dict):
    """Get real-time prices for multiple symbols"""
    try:
        symbols = request.get('symbols', [])
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols list is required")
        
        price_data = await enhanced_market_data_service.get_real_time_prices(symbols)
        
        formatted_prices = {}
        for symbol, data in price_data.items():
            formatted_prices[symbol] = {
                "price": data.get('price'),
                "change": data.get('change_24h'),
                "change_percent": data.get('change_percentage_24h'),
                "volume": data.get('volume_24h'),
                "high_24h": data.get('high_24h'),
                "low_24h": data.get('low_24h'),
                "market_cap": data.get('market_cap'),
                "timestamp": data.get('timestamp')
            }
        
        return {"prices": formatted_prices}
    except Exception as e:
        logger.error(f"Error fetching multiple real-time prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch real-time prices")

@app.get("/api/v2/market-data/overview")
async def get_enhanced_market_overview():
    """Get enhanced market overview with real-time data"""
    try:
        # Get major symbols for overview
        major_symbols = ['BTC', 'ETH', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        price_data = await enhanced_market_data_service.get_real_time_prices(major_symbols)
        
        # Calculate market stats
        total_market_cap = sum(data.get('market_cap', 0) for data in price_data.values())
        total_volume = sum(data.get('volume_24h', 0) for data in price_data.values())
        
        # Sort by change percentage for gainers/losers
        sorted_by_change = sorted(
            price_data.items(),
            key=lambda x: x[1].get('change_percentage_24h', 0),
            reverse=True
        )
        
        return {
            "total_market_cap": total_market_cap,
            "total_volume_24h": total_volume,
            "active_symbols": len(price_data),
            "top_gainers": [
                {
                    "symbol": symbol,
                    "price": data.get('price'),
                    "change_percent": data.get('change_percentage_24h')
                }
                for symbol, data in sorted_by_change[:5]
            ],
            "top_losers": [
                {
                    "symbol": symbol,
                    "price": data.get('price'),
                    "change_percent": data.get('change_percentage_24h')
                }
                for symbol, data in sorted_by_change[-5:]
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching enhanced market overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market overview")

# ============================================================================
# LEGACY MARKET DATA ENDPOINTS
# ============================================================================

@app.get("/market/assets", response_model=List[AssetResponse])
async def get_assets(
    category: Optional[AssetCategory] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get list of available assets"""
    try:
        assets = await market_data_service.get_asset_list(category, limit, offset)
        return assets
    except Exception as e:
        logger.error(f"Error fetching assets: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch assets")

@app.get("/market/asset/{symbol}", response_model=AssetDetailResponse)
async def get_asset_details(symbol: str):
    """Get detailed information about a specific asset"""
    try:
        asset_details = await market_data_service.get_asset_details(symbol)
        return asset_details
    except Exception as e:
        logger.error(f"Error fetching asset details for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Asset {symbol} not found")

@app.get("/market/chart/{symbol}", response_model=ChartDataResponse)
async def get_chart_data(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.ONE_DAY,
    limit: int = 100
):
    """Get chart data for an asset"""
    try:
        chart_data = await market_data_service.get_chart_data(symbol, timeframe, limit)
        return chart_data
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Chart data for {symbol} not found")

@app.get("/market/overview", response_model=MarketOverviewResponse)
async def get_market_overview():
    """Get market overview with top gainers, losers, and trending assets"""
    try:
        overview = await market_data_service.get_market_overview()
        return overview
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market overview")

# ============================================================================
# PORTFOLIO ENDPOINTS
# ============================================================================

@app.get("/portfolio", response_model=List[PortfolioResponse])
async def get_user_portfolios(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all portfolios for the current user"""
    try:
        portfolios = await portfolio_service.get_user_portfolios(current_user.id, db)
        return portfolios
    except Exception as e:
        logger.error(f"Error fetching portfolios for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolios")

@app.post("/portfolio", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    try:
        portfolio = await portfolio_service.create_portfolio(current_user.id, portfolio_data, db)
        return portfolio
    except Exception as e:
        logger.error(f"Error creating portfolio for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create portfolio")

@app.get("/portfolio/{portfolio_id}", response_model=PortfolioDetailResponse)
async def get_portfolio_details(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed portfolio information including performance metrics"""
    try:
        portfolio = await portfolio_service.get_portfolio_details(portfolio_id, current_user.id, db)
        return portfolio
    except Exception as e:
        logger.error(f"Error fetching portfolio {portfolio_id} details: {e}")
        raise HTTPException(status_code=404, detail="Portfolio not found")

@app.post("/portfolio/{portfolio_id}/holdings", response_model=HoldingResponse)
async def add_holding(
    portfolio_id: int,
    holding_data: HoldingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new holding to a portfolio"""
    try:
        holding = await portfolio_service.add_holding(portfolio_id, current_user.id, holding_data, db)
        return holding
    except Exception as e:
        logger.error(f"Error adding holding to portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add holding")

@app.put("/portfolio/{portfolio_id}/holdings/{holding_id}", response_model=HoldingResponse)
async def update_holding(
    portfolio_id: int,
    holding_id: int,
    holding_data: HoldingUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing holding"""
    try:
        holding = await portfolio_service.update_holding(holding_id, current_user.id, holding_data, db)
        return holding
    except Exception as e:
        logger.error(f"Error updating holding {holding_id}: {e}")
        raise HTTPException(status_code=404, detail="Holding not found")

@app.delete("/portfolio/{portfolio_id}/holdings/{holding_id}")
async def delete_holding(
    portfolio_id: int,
    holding_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a holding from a portfolio"""
    try:
        await portfolio_service.delete_holding(holding_id, current_user.id, db)
        return {"message": "Holding deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting holding {holding_id}: {e}")
        raise HTTPException(status_code=404, detail="Holding not found")

@app.get("/portfolio/{portfolio_id}/performance", response_model=PortfolioPerformanceResponse)
async def get_portfolio_performance(
    portfolio_id: int,
    timeframe: TimeFrame = TimeFrame.ONE_MONTH,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio performance metrics"""
    try:
        performance = await portfolio_service.get_portfolio_performance(portfolio_id, current_user.id, timeframe, db)
        return performance
    except Exception as e:
        logger.error(f"Error fetching portfolio {portfolio_id} performance: {e}")
        raise HTTPException(status_code=404, detail="Portfolio not found")

@app.get("/portfolio/{portfolio_id}/rebalance", response_model=RebalanceRecommendationResponse)
async def get_rebalance_recommendations(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio rebalancing recommendations"""
    try:
        recommendations = await portfolio_service.get_rebalance_recommendations(portfolio_id, current_user.id, db)
        return recommendations
    except Exception as e:
        logger.error(f"Error generating rebalance recommendations for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=404, detail="Portfolio not found")

# ============================================================================
# WATCHLIST ENDPOINTS
# ============================================================================

@app.get("/watchlist", response_model=List[WatchlistResponse])
async def get_watchlists(
    current_user: db_models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all watchlists for the current user"""
    watchlists = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).all()
    return [WatchlistResponse(
        id=w.id,
        name=w.name,
        assets=[asset.symbol for asset in w.assets],
        created_at=w.created_at
    ) for w in watchlists]

@app.post("/watchlist", response_model=WatchlistResponse)
async def create_watchlist(
    watchlist_data: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new watchlist"""
    watchlist = Watchlist(name=watchlist_data.name, user_id=current_user.id)
    db.add(watchlist)
    db.commit()
    
    # Add assets to watchlist
    for symbol in watchlist_data.asset_symbols:
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        if asset:
            watchlist.assets.append(asset)
    
    db.commit()
    db.refresh(watchlist)
    
    return WatchlistResponse(
        id=watchlist.id,
        name=watchlist.name,
        assets=[asset.symbol for asset in watchlist.assets],
        created_at=watchlist.created_at
    )

@app.delete("/watchlist/{watchlist_id}")
async def delete_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a watchlist"""
    watchlist = db.query(Watchlist).filter(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id
    ).first()
    
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    
    db.delete(watchlist)
    db.commit()
    
    return {"message": "Watchlist deleted successfully"}

# ============================================================================
# TRADING ENDPOINTS
# ============================================================================

@app.post("/trading/order", response_model=OrderResponse)
async def create_order(
    order_data: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new trading order"""
    try:
        order = await trading_service.create_order(current_user.id, order_data, db)
        return order
    except Exception as e:
        logger.error(f"Error creating order for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create order")

@app.get("/trading/orders", response_model=List[OrderResponse])
async def get_user_orders(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's trading orders"""
    try:
        orders = await trading_service.get_user_orders(current_user.id, status, limit, offset, db)
        return orders
    except Exception as e:
        logger.error(f"Error fetching orders for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch orders")

@app.delete("/trading/order/{order_id}")
async def cancel_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel a trading order"""
    try:
        await trading_service.cancel_order(order_id, current_user.id, db)
        return {"message": "Order cancelled successfully"}
    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e}")
        raise HTTPException(status_code=404, detail="Order not found")

@app.get("/trading/strategies", response_model=List[TradingStrategyResponse])
async def get_trading_strategies(
    current_user: User = Depends(get_current_user)
):
    """Get available trading strategies"""
    try:
        strategies = await trading_service.get_available_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error fetching trading strategies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch strategies")

@app.post("/trading/strategy/execute", response_model=StrategyExecutionResponse)
async def execute_strategy(
    strategy_data: StrategyExecutionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Execute a trading strategy"""
    try:
        result = await trading_service.execute_strategy(current_user.id, strategy_data, db)
        return result
    except Exception as e:
        logger.error(f"Error executing strategy for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute strategy")

# ============================================================================
# NEWS ENDPOINTS
# ============================================================================

@app.get("/news", response_model=List[NewsResponse])
async def get_news(
    category: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """Get financial news articles"""
    try:
        news = await news_service.get_news(category=category, limit=limit, offset=offset)
        return news
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch news")

@app.get("/news/{article_id}", response_model=NewsDetailResponse)
async def get_news_article(article_id: str):
    """Get a specific news article"""
    try:
        article = await news_service.get_article(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {e}")
        raise HTTPException(status_code=404, detail="Article not found")

@app.get("/news/sentiment/{symbol}", response_model=NewsSentimentResponse)
async def get_news_sentiment(symbol: str):
    """Get news sentiment analysis for a specific asset"""
    try:
        sentiment = await news_service.get_sentiment_analysis(symbol)
        return sentiment
    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sentiment")

# ============================================================================
# FORUM ENDPOINTS
# ============================================================================

@app.get("/forum/posts", response_model=List[ForumPostResponse])
async def get_forum_posts(
    category: Optional[ForumCategory] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get forum posts"""
    try:
        posts = await forum_service.get_posts(category, limit, offset, db)
        return posts
    except Exception as e:
        logger.error(f"Error fetching forum posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch posts")

@app.post("/forum/posts", response_model=ForumPostResponse)
async def create_forum_post(
    post_data: ForumPostCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new forum post"""
    try:
        post = await forum_service.create_post(current_user.id, post_data, db)
        return post
    except Exception as e:
        logger.error(f"Error creating forum post for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create post")

@app.get("/forum/posts/{post_id}", response_model=ForumPostDetailResponse)
async def get_forum_post(
    post_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific forum post with comments"""
    try:
        post = await forum_service.get_post_with_comments(post_id, db)
        return post
    except Exception as e:
        logger.error(f"Error fetching forum post {post_id}: {e}")
        raise HTTPException(status_code=404, detail="Post not found")

@app.post("/forum/posts/{post_id}/comments", response_model=CommentResponse)
async def create_comment(
    post_id: int,
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a comment on a forum post"""
    try:
        comment = await forum_service.create_comment(post_id, current_user.id, comment_data, db)
        return comment
    except Exception as e:
        logger.error(f"Error creating comment for post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create comment")

@app.post("/forum/posts/{post_id}/vote")
async def vote_on_post(
    post_id: int,
    vote_data: VoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Vote on a forum post"""
    try:
        await forum_service.vote_on_post(post_id, current_user.id, vote_data.is_upvote, db)
        return {"message": "Vote recorded successfully"}
    except Exception as e:
        logger.error(f"Error voting on post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to record vote")

@app.get("/forum/trending", response_model=List[TrendingTopicResponse])
async def get_trending_topics(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get trending forum topics"""
    try:
        topics = await forum_service.get_trending_topics(limit, db)
        return topics
    except Exception as e:
        logger.error(f"Error fetching trending topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trending topics")

# ============================================================================
# AI ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/ai/analyze", response_model=AIAnalysisResponse)
async def analyze_asset(
    analysis_request: AIAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Get AI analysis for an asset"""
    try:
        analysis = await ai_service.analyze_asset(
            analysis_request.symbol, 
            analysis_request.analysis_type
        )
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing asset {analysis_request.symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze asset")

@app.post("/ai/explain", response_model=AIExplanationResponse)
async def explain_market_data(
    explanation_request: AIExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Get AI explanation for market data"""
    try:
        explanation = await ai_service.explain_data(
            explanation_request.data, 
            explanation_request.context
        )
        return AIExplanationResponse(explanation=explanation)
    except Exception as e:
        logger.error(f"Error explaining market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to explain data")

@app.get("/ai/insights", response_model=List[AIInsightResponse])
async def get_ai_insights(
    symbols: Optional[List[str]] = None,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Get AI-generated market insights"""
    try:
        insights = await ai_service.get_insights(symbols=symbols, limit=limit)
        return insights
    except Exception as e:
        logger.error(f"Error fetching AI insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch insights")

@app.post("/ai/portfolio/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    optimization_request: PortfolioOptimizationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI-powered portfolio optimization recommendations"""
    try:
        optimization = await ai_service.optimize_portfolio(
            current_user.id,
            optimization_request,
            db
        )
        return optimization
    except Exception as e:
        logger.error(f"Error optimizing portfolio for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize portfolio")

@app.get("/ai/predictions/{symbol}", response_model=PricePredictionResponse)
async def get_price_predictions(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.ONE_WEEK,
    current_user: User = Depends(get_current_user)
):
    """Get AI price predictions for an asset"""
    try:
        predictions = await ai_service.get_price_predictions(symbol, timeframe)
        return predictions
    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")

# ============================================================================
# NOTIFICATION ENDPOINTS
# ============================================================================

@app.get("/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    try:
        notifications = await notification_service.get_user_notifications(
            current_user.id, unread_only, limit, offset, db
        )
        return notifications
    except Exception as e:
        logger.error(f"Error fetching notifications for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications")

@app.put("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    try:
        await notification_service.mark_as_read(notification_id, current_user.id, db)
        return {"message": "Notification marked as read"}
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {e}")
        raise HTTPException(status_code=404, detail="Notification not found")

@app.post("/notifications/preferences", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    preferences: NotificationPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user notification preferences"""
    try:
        updated_preferences = await notification_service.update_preferences(
            current_user.id, preferences, db
        )
        return updated_preferences
    except Exception as e:
        logger.error(f"Error updating notification preferences for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Send real-time market data
            market_data = await market_data_service.get_real_time_data()
            await ws_manager.send_personal_message(
                json.dumps({
                    "type": "market_update",
                    "data": market_data,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            await asyncio.sleep(5)  # Update every 5 seconds
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/portfolio/{portfolio_id}")
async def portfolio_websocket(
    websocket: WebSocket,
    portfolio_id: int
):
    """WebSocket endpoint for real-time portfolio updates"""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Send portfolio performance updates
            portfolio_data = await portfolio_service.get_real_time_performance(portfolio_id)
            await ws_manager.send_personal_message(
                json.dumps({
                    "type": "portfolio_update",
                    "portfolio_id": portfolio_id,
                    "data": portfolio_data,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            await asyncio.sleep(10)  # Update every 10 seconds
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/notifications")
async def notifications_websocket(
    websocket: WebSocket,
    token: str
):
    """WebSocket endpoint for real-time notifications"""
    # Verify user token
    try:
        user = await get_current_user_from_token(token)
        await ws_manager.connect(websocket, user.id)
        
        while True:
            # Check for new notifications
            notifications = await notification_service.get_pending_notifications(user.id)
            if notifications:
                await ws_manager.send_personal_message(
                    json.dumps({
                        "type": "notifications",
                        "data": notifications,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
            await asyncio.sleep(30)  # Check every 30 seconds
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close()
    except WebSocketDisconnect:
        if 'user' in locals():
            ws_manager.disconnect(websocket, user.id)

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

# ============================================================================
# PHASE 6: LLM EXPLANATION ENGINE ENDPOINTS
# ============================================================================

@app.post("/api/llm/explain", response_model=ExplanationResponse)
async def generate_explanation(
    request: ExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate LLM explanation"""
    try:
        response = await llm_service.generate_explanation(request)
        return response
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/explain/stream")
async def stream_explanation(
    request: ExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Stream LLM explanation"""
    try:
        async for chunk in llm_service.stream_explanation(request):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:
        logger.error(f"Error streaming explanation: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

@app.post("/api/financial/explain", response_model=FinancialExplanationResponse)
async def generate_financial_explanation(
    request: FinancialExplanationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive financial explanation"""
    try:
        # Set user context
        request.user_id = str(current_user.id)
        
        response = await explanation_generator.generate_explanation(request)
        return response
    except Exception as e:
        logger.error(f"Error generating financial explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation/create", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest,
    current_user: User = Depends(get_current_user)
):
    """Create new conversation"""
    try:
        # Set user context
        request.user_id = str(current_user.id)
        
        response = await conversation_manager.create_conversation(request)
        return response
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation/{conversation_id}/message", response_model=ConversationResponse)
async def add_conversation_message(
    conversation_id: str,
    request: ConversationRequest,
    current_user: User = Depends(get_current_user)
):
    """Add message to conversation"""
    try:
        response = await conversation_manager.add_message(
            conversation_id, request.message, request.context
        )
        return response
    except Exception as e:
        logger.error(f"Error adding conversation message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get conversation by ID"""
    try:
        conversation = await conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_user_conversations(
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get user conversations"""
    try:
        conversations = await conversation_manager.get_user_conversations(
            str(current_user.id), limit, offset
        )
        return conversations
    except Exception as e:
        logger.error(f"Error getting user conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if LANGCHAIN_INTEGRATION_AVAILABLE:
    @app.post("/api/langchain/process", response_model=LangChainResponse)
    async def process_langchain_request(
        request: LangChainRequest,
        current_user: User = Depends(get_current_user)
    ):
        """Process request using LangChain"""
        try:
            if not langchain_integration:
                raise HTTPException(status_code=503, detail="LangChain integration not available")
            
            # Set user context
            request.user_id = str(current_user.id)
            
            response = await langchain_integration.process_request(request)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing LangChain request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/context/{symbol}")
async def get_financial_context(
    symbol: str,
    context_types: Optional[str] = None,
    timeframe: str = "1d",
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive financial context for a symbol"""
    try:
        # Parse context types
        types = None
        if context_types:
            types = [ContextType(t.strip()) for t in context_types.split(",")]
        
        context = await financial_context_engine.get_comprehensive_context(
            symbol=symbol,
            context_types=types,
            timeframe=timeframe,
            user_id=str(current_user.id)
        )
        return context
    except Exception as e:
        logger.error(f"Error getting financial context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llm/models")
async def get_available_models(
    current_user: User = Depends(get_current_user)
):
    """Get available LLM models and capabilities"""
    try:
        models = await llm_service.get_available_models()
        
        response = {
            "llm_models": models,
            "prompt_types": [pt.value for pt in PromptType],
            "explanation_complexity": [ec.value for ec in ExplanationComplexity],
            "providers": [p.value for p in LLMProvider]
        }
        
        if langchain_integration:
            langchain_models = langchain_integration.get_available_models()
            response["langchain"] = langchain_models
        
        return response
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/llm/status")
async def get_llm_service_status(
    current_user: User = Depends(get_current_user)
):
    """Get LLM service status"""
    try:
        status = await llm_service.get_service_status()
        
        if langchain_integration:
            langchain_status = langchain_integration.get_service_status()
            status["langchain"] = langchain_status
        
        return status
    except Exception as e:
        logger.error(f"Error getting LLM service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "database": "connected",
            "market_data": "active",
            "ai_service": "active",
            "websocket": "active"
        }
    }

# Enhanced market data endpoints (Phase 3)
@app.get("/api/v2/assets")
async def get_enhanced_assets(
    category: Optional[str] = None,
    limit: int = 50,
    force_refresh: bool = False
):
    """Get assets with enhanced caching and multiple data sources."""
    try:
        assets = await enhanced_market_data_service.get_assets(
            category=category,
            limit=limit,
            force_refresh=force_refresh
        )
        return {"assets": [asset.dict() for asset in assets]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/prices/realtime")
async def get_real_time_prices(symbols: str):
    """Get real-time prices for multiple symbols."""
    try:
        symbol_list = symbols.split(',')
        prices = await enhanced_market_data_service.get_real_time_prices(symbol_list)
        return {"prices": prices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/streaming/status")
async def get_streaming_status():
    """Get real-time streaming service status."""
    try:
        status = await real_time_streaming_service.get_active_subscriptions()
        return {"streaming_status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Phase 4: News & Sentiment Engine API Endpoints
@app.get("/api/v2/news")
async def get_enhanced_news(
    categories: Optional[str] = None,
    limit: int = 50,
    include_analysis: bool = True
):
    """Get enhanced news with sentiment analysis."""
    try:
        service = await get_enhanced_news_service()
        category_list = [c.strip() for c in categories.split(",")] if categories else None
        articles = await service.fetch_latest_news(
            categories=category_list,
            limit=limit,
            include_analysis=include_analysis
        )
        return {
            "articles": [article.__dict__ for article in articles],
            "count": len(articles),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching enhanced news: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch enhanced news")

@app.get("/api/v2/news/search")
async def search_news(
    query: str,
    categories: Optional[str] = None,
    symbols: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 20
):
    """Search news articles with advanced filtering."""
    try:
        from enhanced_news_service import NewsFilter
        service = await get_enhanced_news_service()
        
        # Build filters
        filters = NewsFilter(
            categories=[c.strip() for c in categories.split(",")] if categories else None,
            symbols=[s.strip() for s in symbols.split(",")] if symbols else None,
            sentiment=[s.strip() for s in sentiment.split(",")] if sentiment else None
        )
        
        articles = await service.search_news(query, filters, limit)
        return {
            "articles": [article.__dict__ for article in articles],
            "count": len(articles),
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error searching news: {e}")
        raise HTTPException(status_code=500, detail="Failed to search news")

@app.get("/api/v2/news/trending")
async def get_trending_topics(hours: int = 24):
    """Get trending topics and keywords."""
    try:
        service = await get_enhanced_news_service()
        trends = await service.get_trending_topics(hours)
        return {
            "trends": trends,
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting trending topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trending topics")

@app.get("/api/v2/sentiment/overview")
async def get_sentiment_overview(
    categories: Optional[str] = None
):
    """Get overall sentiment analysis for news."""
    try:
        service = await get_enhanced_news_service()
        category_list = [c.strip() for c in categories.split(",")] if categories else None
        overview = await service.get_sentiment_overview(category_list)
        return {
            "sentiment_overview": overview,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sentiment overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sentiment overview")

@app.post("/api/v2/sentiment/analyze")
async def analyze_sentiment(request: dict):
    """Analyze sentiment of provided text."""
    try:
        service = await get_sentiment_engine()
        title = request.get("title", "")
        content = request.get("content", "")
        url = request.get("url", "")
        
        if not title and not content:
            raise HTTPException(status_code=400, detail="Title or content is required")
        
        analysis = await service.analyze_news_article(title, content, url)
        return {
            "analysis": analysis.__dict__,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze sentiment")

@app.post("/api/v2/content/filter")
async def filter_content(request: dict):
    """Filter and analyze content quality."""
    try:
        service = await get_content_filter_service()
        title = request.get("title", "")
        content = request.get("content", "")
        url = request.get("url", "")
        source = request.get("source", "")
        author = request.get("author", "")
        
        if not title and not content:
            raise HTTPException(status_code=400, detail="Title or content is required")
        
        result = await service.filter_content(title, content, url, source, author)
        return {
            "filter_result": {
                "action": result.action.value,
                "category": result.category.value,
                "quality": result.quality.value,
                "scores": {
                    "quality_score": result.scores.quality_score,
                    "relevance_score": result.scores.relevance_score,
                    "credibility_score": result.scores.credibility_score,
                    "readability_score": result.scores.readability_score,
                    "engagement_score": result.scores.engagement_score,
                    "spam_score": result.scores.spam_score,
                    "overall_score": result.scores.overall_score
                },
                "flags": result.flags,
                "reasons": result.reasons,
                "confidence": result.confidence,
                "processing_time": result.processing_time
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error filtering content: {e}")
        raise HTTPException(status_code=500, detail="Failed to filter content")

@app.get("/api/v2/content/metrics")
async def get_content_metrics(text: str):
    """Get detailed content metrics."""
    try:
        service = await get_content_filter_service()
        metrics = await service.get_content_metrics(text)
        return {
            "metrics": {
                "word_count": metrics.word_count,
                "sentence_count": metrics.sentence_count,
                "paragraph_count": metrics.paragraph_count,
                "reading_time": metrics.reading_time,
                "complexity_score": metrics.complexity_score,
                "keyword_density": metrics.keyword_density,
                "entity_count": metrics.entity_count,
                "link_count": metrics.link_count,
                "image_count": metrics.image_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting content metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get content metrics")

# WebSocket endpoint for real-time data streaming
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(
    websocket: WebSocket,
    symbols: Optional[str] = None,
    stream_types: Optional[str] = None,
    user_id: Optional[str] = None,
    portfolio_id: Optional[str] = None
):
    """WebSocket endpoint for real-time data streaming."""
    import uuid
    client_id = str(uuid.uuid4())
    
    try:
        # Parse parameters
        symbol_list = symbols.split(',') if symbols else []
        stream_type_list = stream_types.split(',') if stream_types else []
        
        # Subscribe client to real-time streams
        await real_time_streaming_service.subscribe_client(
            websocket=websocket,
            client_id=client_id,
            symbols=symbol_list,
            stream_types=stream_type_list,
            user_id=user_id,
            portfolio_id=portfolio_id
        )
        
    except WebSocketDisconnect:
        await real_time_streaming_service.unsubscribe_client(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await real_time_streaming_service.unsubscribe_client(client_id)

# Data normalization endpoints
@app.post("/api/data/normalize")
async def normalize_data(data: dict, source: str):
    """Normalize data from external sources."""
    try:
        from data_normalization import DataSource
        
        # Convert string to DataSource enum
        data_source = DataSource(source)
        
        normalized_asset = data_normalization_service.normalize_asset_data(
            raw_data=data,
            source=data_source
        )
        
        if normalized_asset:
            return {"normalized_data": normalized_asset.to_asset_response().dict()}
        else:
            raise HTTPException(status_code=400, detail="Failed to normalize data")
            
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported data source: {source}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 7: ADVANCED FEATURES ENDPOINTS
# ============================================================================

# Portfolio Management Endpoints
@app.post("/api/v2/portfolio/create")
async def create_portfolio_v2(
    request: PortfolioRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    try:
        portfolio = await portfolio_manager.create_portfolio(request, current_user.id, db)
        return portfolio
    except Exception as e:
        logger.error(f"Error creating portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/portfolio/{portfolio_id}")
async def get_portfolio_v2(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio details"""
    try:
        portfolio = await portfolio_manager.get_portfolio(portfolio_id, current_user.id, db)
        return portfolio
    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/portfolio/{portfolio_id}/transaction")
async def add_transaction(
    portfolio_id: str,
    request: TransactionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add transaction to portfolio"""
    try:
        result = await portfolio_manager.add_transaction(portfolio_id, request, current_user.id, db)
        return result
    except Exception as e:
        logger.error(f"Error adding transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/portfolio/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: str,
    request: RebalanceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rebalance portfolio"""
    try:
        result = await portfolio_manager.rebalance_portfolio(portfolio_id, request, current_user.id, db)
        return result
    except Exception as e:
        logger.error(f"Error rebalancing portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Portfolio Analytics Endpoints
@app.get("/api/v2/portfolio/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio performance metrics"""
    try:
        performance = await portfolio_analytics.calculate_performance_metrics(portfolio_id, db)
        return performance
    except Exception as e:
        logger.error(f"Error calculating performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/portfolio/{portfolio_id}/risk")
async def get_portfolio_risk(
    portfolio_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get portfolio risk metrics"""
    try:
        risk_metrics = await portfolio_analytics.calculate_risk_metrics(portfolio_id, db)
        return risk_metrics
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Risk Management Endpoints
@app.post("/api/v2/risk/assess")
async def assess_risk(
    request: RiskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Assess portfolio risk"""
    try:
        assessment = await risk_engine.assess_portfolio_risk(request, db)
        return assessment
    except Exception as e:
        logger.error(f"Error assessing risk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/risk/stress-test")
async def stress_test(
    request: RiskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform stress test"""
    try:
        # Get portfolio holdings first
        portfolio = db.query(Portfolio).filter(Portfolio.id == request.portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        holdings = db.query(PortfolioHolding).filter(PortfolioHolding.portfolio_id == request.portfolio_id).all()
        holdings_data = [{
            "symbol": h.symbol,
            "quantity": h.quantity,
            "current_price": h.current_price,
            "market_value": h.market_value
        } for h in holdings]
        
        # Ensure stress tests are included
        stress_request = RiskRequest(
            portfolio_id=request.portfolio_id,
            risk_horizon_days=request.risk_horizon_days if hasattr(request, 'risk_horizon_days') else 30,
            confidence_level=request.confidence_level if hasattr(request, 'confidence_level') else 0.95,
            include_stress_tests=True,
            custom_scenarios=request.custom_scenarios if hasattr(request, 'custom_scenarios') else None
        )
        
        results = await risk_engine.assess_portfolio_risk(stress_request, holdings_data, db)
        return results.assessment.stress_test_results
    except Exception as e:
        logger.error(f"Error running stress test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/risk/limits")
async def set_risk_limits(
    request: RiskLimitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Set risk limits"""
    try:
        result = await risk_engine.set_risk_limits(request, current_user.id, db)
        return result
    except Exception as e:
        logger.error(f"Error setting risk limits: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading Engine Endpoints
@app.post("/api/v2/trading/order")
async def submit_order(
    request: OrderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit trading order"""
    try:
        order = await trading_engine.submit_order(request, current_user.id, db)
        return order
    except Exception as e:
        logger.error(f"Error submitting order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/trading/orders")
async def get_orders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user orders"""
    try:
        orders = await trading_engine.get_user_orders(current_user.id, db)
        return orders
    except Exception as e:
        logger.error(f"Error getting orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v2/trading/order/{order_id}")
async def cancel_order(
    order_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel trading order"""
    try:
        result = await trading_engine.cancel_order(order_id, current_user.id, db)
        return result
    except Exception as e:
        logger.error(f"Error canceling order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/trading/execute")
async def execute_strategy(
    request: TradingRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Execute trading strategy"""
    try:
        result = await trading_engine.execute_trading_strategy(request, current_user.id, db)
        return result
    except Exception as e:
        logger.error(f"Error executing strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Community Forum Endpoints
@app.get("/api/v2/forum/posts")
async def get_forum_posts(
    skip: int = 0,
    limit: int = 20,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get forum posts"""
    try:
        posts = await community_forum.get_posts(skip, limit, category, db)
        return posts
    except Exception as e:
        logger.error(f"Error getting forum posts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/forum/posts")
async def create_forum_post(
    request: PostRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create forum post"""
    try:
        post = await community_forum.create_post(request, current_user.id, db)
        return post
    except Exception as e:
        logger.error(f"Error creating forum post: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/forum/posts/{post_id}/comments")
async def add_comment(
    post_id: str,
    request: CommentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add comment to post"""
    try:
        comment = await community_forum.add_comment(post_id, request, current_user.id, db)
        return comment
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/forum/trending")
async def get_trending_posts(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get trending forum posts"""
    try:
        posts = await community_forum.get_trending_posts(limit, db)
        return posts
    except Exception as e:
        logger.error(f"Error getting trending posts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Notification System Endpoints
@app.post("/api/v2/notifications/alert")
async def create_alert(
    request: AlertRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create price alert"""
    try:
        alert = await notification_system.create_price_alert(request, current_user.id, db)
        return alert
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/notifications")
async def get_notifications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    try:
        notifications = await notification_system.get_user_notifications(current_user.id, db)
        return notifications
    except Exception as e:
        logger.error(f"Error getting notifications: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/notifications/send")
async def send_notification(
    request: NotificationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send notification"""
    try:
        result = await notification_system.send_notification(request, db)
        return result
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Charting Endpoints
@app.post("/api/v2/charts/data")
async def get_chart_data(
    request: ChartRequest,
    db: Session = Depends(get_db)
):
    """Get advanced chart data with indicators and patterns"""
    try:
        chart_data = await advanced_charting.get_chart_data(request, db)
        return chart_data
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/charts/indicator")
async def calculate_indicator(
    request: IndicatorRequest,
    db: Session = Depends(get_db)
):
    """Calculate technical indicator"""
    try:
        indicator_data = await advanced_charting.calculate_indicator(request, db)
        return indicator_data
    except Exception as e:
        logger.error(f"Error calculating indicator: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/charts/patterns")
async def scan_patterns(
    request: PatternScanRequest,
    db: Session = Depends(get_db)
):
    """Scan for chart patterns"""
    try:
        patterns = await advanced_charting.scan_patterns(request, db)
        return patterns
    except Exception as e:
        logger.error(f"Error scanning patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def api_status():
    """Detailed API status"""
    try:
        # Check database connection
        db_status = await check_database_connection()
        
        # Check external services
        market_status = await market_data_service.health_check()
        ai_status = await ai_service.health_check()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": db_status,
                "market_data_service": market_status,
                "ai_service": ai_status,
                "websocket_manager": "operational",
                "phase_7_services": {
                    "portfolio_manager": "operational",
                    "risk_engine": "operational",
                    "trading_engine": "operational",
                    "community_forum": "operational",
                    "notification_system": "operational",
                    "advanced_charting": "operational"
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# ============================================================================
# AI-FIRST PLATFORM ENDPOINTS
# ============================================================================

@app.get("/ai/status")
async def get_ai_status():
    """Get AI core system status"""
    if not ai_core_engine:
        return {"status": "unavailable", "message": "AI core not initialized"}
    
    return ai_core_engine.get_system_status()

@app.get("/ai/features")
async def get_ai_features():
    """Get available AI features"""
    if not ai_core_engine:
        return {"features": [], "message": "AI core not initialized"}
    
    return {"features": ai_core_engine.get_available_features()}

@app.post("/ai/personalize")
async def personalize_experience(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Get personalized experience for user"""
    if not ai_core_engine:
        raise HTTPException(status_code=503, detail="AI core not available")
    
    try:
        result = await ai_core_engine.process_user_request(
            user_id=current_user.id,
            request_type="personalization",
            data=request
        )
        return result
    except Exception as e:
        logger.error(f"Personalization error: {e}")
        raise HTTPException(status_code=500, detail="Personalization failed")

@app.post("/ai/trading/analyze")
async def analyze_trading_opportunity(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Analyze trading opportunity using AI"""
    if not ai_core_engine:
        raise HTTPException(status_code=503, detail="AI core not available")
    
    try:
        result = await ai_core_engine.process_user_request(
            user_id=current_user.id,
            request_type="trading_analysis",
            data=request
        )
        return result
    except Exception as e:
        logger.error(f"Trading analysis error: {e}")
        raise HTTPException(status_code=500, detail="Trading analysis failed")

@app.post("/ai/chat")
async def ai_chat(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Chat with AI assistant"""
    if not ai_core_engine:
        raise HTTPException(status_code=503, detail="AI core not available")
    
    try:
        result = await ai_core_engine.process_user_request(
            user_id=current_user.id,
            request_type="natural_language",
            data=request
        )
        return result
    except Exception as e:
        logger.error(f"AI chat error: {e}")
        raise HTTPException(status_code=500, detail="AI chat failed")

@app.post("/ai/predict")
async def ai_predict(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Get AI predictions"""
    if not ai_core_engine:
        raise HTTPException(status_code=503, detail="AI core not available")
    
    try:
        result = await ai_core_engine.process_user_request(
            user_id=current_user.id,
            request_type="prediction",
            data=request
        )
        return result
    except Exception as e:
        logger.error(f"AI prediction error: {e}")
        raise HTTPException(status_code=500, detail="AI prediction failed")

# ============================================================================
# DEFI PLATFORM ENDPOINTS
# ============================================================================

@app.get("/defi/status")
async def get_defi_status():
    """Get DeFi core system status"""
    if not defi_core_engine:
        return {"status": "unavailable", "message": "DeFi core not initialized"}
    
    return defi_core_engine.get_status()

@app.get("/defi/features")
async def get_defi_features():
    """Get available DeFi features"""
    if not defi_core_engine:
        return {"features": [], "message": "DeFi core not initialized"}
    
    return {"features": defi_core_engine.get_features()}

@app.post("/defi/protocols")
async def get_supported_protocols(
    current_user: User = Depends(get_current_user)
):
    """Get supported DeFi protocols"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="protocols",
            data={"action": "list_supported"}
        )
        return {"status": "success", "protocols": result}
    except Exception as e:
        logger.error(f"Protocol listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get protocols")

@app.post("/defi/yield/optimize")
async def optimize_yield(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Optimize yield farming strategies"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="yield_optimization",
            data=request
        )
        return {"status": "success", "optimization": result}
    except Exception as e:
        logger.error(f"Yield optimization error: {e}")
        raise HTTPException(status_code=500, detail="Yield optimization failed")

@app.post("/defi/liquidity/manage")
async def manage_liquidity(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Manage liquidity positions"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="liquidity_management",
            data=request
        )
        return {"status": "success", "liquidity": result}
    except Exception as e:
        logger.error(f"Liquidity management error: {e}")
        raise HTTPException(status_code=500, detail="Liquidity management failed")

@app.post("/defi/bridge/transfer")
async def cross_chain_transfer(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Execute cross-chain asset transfer"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="cross_chain_bridge",
            data=request
        )
        return {"status": "success", "transfer": result}
    except Exception as e:
        logger.error(f"Cross-chain transfer error: {e}")
        raise HTTPException(status_code=500, detail="Cross-chain transfer failed")

@app.post("/defi/gas/optimize")
async def optimize_gas(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Optimize gas fees for transactions"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="gas_optimization",
            data=request
        )
        return {"status": "success", "gas_optimization": result}
    except Exception as e:
        logger.error(f"Gas optimization error: {e}")
        raise HTTPException(status_code=500, detail="Gas optimization failed")

@app.post("/defi/nft/analyze")
async def analyze_nft(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Analyze NFT investments and market trends"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="nft_analysis",
            data=request
        )
        return {"status": "success", "nft_analysis": result}
    except Exception as e:
        logger.error(f"NFT analysis error: {e}")
        raise HTTPException(status_code=500, detail="NFT analysis failed")

@app.post("/defi/analytics/blockchain")
async def blockchain_analytics(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Get blockchain analytics and insights"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="blockchain_analytics",
            data=request
        )
        return {"status": "success", "analytics": result}
    except Exception as e:
        logger.error(f"Blockchain analytics error: {e}")
        raise HTTPException(status_code=500, detail="Blockchain analytics failed")

@app.post("/defi/identity/create")
async def create_decentralized_identity(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Create decentralized identity (DID)"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="decentralized_identity",
            data={**request, "action": "create_did"}
        )
        return {"status": "success", "identity": result}
    except Exception as e:
        logger.error(f"DID creation error: {e}")
        raise HTTPException(status_code=500, detail="DID creation failed")

@app.post("/defi/identity/authenticate")
async def authenticate_with_did(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Authenticate using decentralized identity"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="decentralized_identity",
            data={**request, "action": "authenticate"}
        )
        return {"status": "success", "authentication": result}
    except Exception as e:
        logger.error(f"DID authentication error: {e}")
        raise HTTPException(status_code=500, detail="DID authentication failed")

@app.post("/defi/identity/credentials")
async def manage_credentials(
    request: dict,
    current_user: User = Depends(get_current_user)
):
    """Manage verifiable credentials"""
    if not defi_core_engine:
        raise HTTPException(status_code=503, detail="DeFi core not available")
    
    try:
        result = await defi_core_engine.process_request(
            user_id=str(current_user.id),
            request_type="decentralized_identity",
            data={**request, "action": "manage_credentials"}
        )
        return {"status": "success", "credentials": result}
    except Exception as e:
        logger.error(f"Credential management error: {e}")
        raise HTTPException(status_code=500, detail="Credential management failed")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def get_current_user_from_token(token: str) -> User:
    """Get user from WebSocket token"""
    try:
        from auth import verify_token
        payload = verify_token(token)
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        from database import SessionLocal
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            return user
        finally:
            db.close()
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

async def check_database_connection() -> str:
    """Check database connection status"""
    try:
        from database import SessionLocal
        db = SessionLocal()
        try:
            # Simple query to test connection
            db.execute("SELECT 1")
            return "connected"
        finally:
            db.close()
    except Exception:
        return "disconnected"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )