"""FinScope FastAPI Application - Refined Architecture.

This is the main application file that demonstrates the refined architecture
with proper dependency injection, feature loading, and modular design.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import configuration and core components
from app.config.settings import get_settings
from app.core.feature_registry import registry, register_all_features
from app.core.exceptions import FinScopeException
from app.core.middleware import setup_middleware
from app.core.dependencies import get_db, get_current_user

# Import API routers
from app.api.router import api_router

# Import core services (always available)
from database import init_db
from websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Load features based on configuration
    logger.info("Loading features...")
    feature_config = settings.get_feature_config()
    load_results = registry.load_all_features(feature_config)
    
    # Log feature loading results
    for feature_name, success in load_results.items():
        if success:
            logger.info(f"✓ Feature '{feature_name}' loaded successfully")
        else:
            status = registry.get_status(feature_name)
            logger.warning(f"✗ Feature '{feature_name}' failed to load: {status}")
    
    # Store loaded features in app state
    app.state.features = registry.get_available_features()
    app.state.feature_summary = registry.get_feature_summary()
    
    # Initialize WebSocket manager
    app.state.websocket_manager = WebSocketManager()
    
    logger.info(f"{settings.app_name} startup complete")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")
    
    # Cleanup WebSocket connections
    if hasattr(app.state, 'websocket_manager'):
        await app.state.websocket_manager.disconnect_all()
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced Financial Analysis and Trading Platform",
    docs_url=settings.docs_url if not settings.is_production else None,
    redoc_url=settings.redoc_url if not settings.is_production else None,
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app, settings)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# Global exception handlers
@app.exception_handler(FinScopeException)
async def finscope_exception_handler(request: Request, exc: FinScopeException):
    """Handle FinScope-specific exceptions."""
    logger.error(f"FinScope exception: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "type": exc.__class__.__name__,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if settings.is_production:
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "type": exc.__class__.__name__,
                "path": str(request.url.path)
            }
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


# Features status endpoint
@app.get("/features")
async def features_status():
    """Get status of all features."""
    return {
        "features": registry.get_feature_summary(),
        "available_count": len(registry.get_available_features()),
        "total_count": len(registry.get_all_features())
    }


# Include API routers
app.include_router(api_router, prefix=settings.api_v1_prefix)


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time communication."""
    websocket_manager = app.state.websocket_manager
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Process incoming WebSocket messages here
            await websocket_manager.send_personal_message(
                f"Echo: {data}", websocket
            )
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=settings.workers if settings.is_production else 1,
        log_level=settings.monitoring.log_level.lower()
    )