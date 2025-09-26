from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel as PydanticBaseModel
import pandas as pd
import json
from datetime import datetime

from ..models.model_factory import model_factory
from ..models.base_model import BaseModel

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(PydanticBaseModel):
    model_id: str
    name: str
    category: str
    model_type: str
    description: str
    status: str
    accuracy: Optional[float] = None
    last_run: Optional[str] = None


class TrainModelRequest(PydanticBaseModel):
    model_id: str
    data: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = {}


class PredictRequest(PydanticBaseModel):
    model_id: str
    data: List[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]] = {}


class ModelResponse(PydanticBaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@router.get("/", response_model=List[ModelInfo])
async def get_all_models():
    """
    Get information about all available models.
    """
    try:
        models = model_factory.get_all_models()
        model_infos = []
        
        for model in models:
            info = model.get_info()
            model_info = ModelInfo(
                model_id=info['model_id'],
                name=info['name'],
                category=info['category'],
                model_type=info['model_type'],
                description=info['description'],
                status=info['status'],
                accuracy=info.get('accuracy'),
                last_run=info.get('last_run')
            )
            model_infos.append(model_info)
        
        return model_infos
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/category/{category}", response_model=List[ModelInfo])
async def get_models_by_category(category: str):
    """
    Get models by category (Crypto, Stock, Forex, Futures, Index, Cross-Asset).
    """
    try:
        models = model_factory.get_models_by_category(category)
        model_infos = []
        
        for model in models:
            info = model.get_info()
            model_info = ModelInfo(
                model_id=info['model_id'],
                name=info['name'],
                category=info['category'],
                model_type=info['model_type'],
                description=info['description'],
                status=info['status'],
                accuracy=info.get('accuracy'),
                last_run=info.get('last_run')
            )
            model_infos.append(model_info)
        
        return model_infos
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """
    Get information about a specific model.
    """
    try:
        model = model_factory.get_model(model_id)
        info = model.get_info()
        
        return ModelInfo(
            model_id=info['model_id'],
            name=info['name'],
            category=info['category'],
            model_type=info['model_type'],
            description=info['description'],
            status=info['status'],
            accuracy=info.get('accuracy'),
            last_run=info.get('last_run')
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=ModelResponse)
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Train a model with provided data.
    """
    try:
        model = model_factory.get_model(request.model_id)
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Train model in background
        def train_task():
            return model.train(df, **request.parameters)
        
        # For now, train synchronously (in production, use background tasks)
        result = train_task()
        
        return ModelResponse(
            status="success",
            message=f"Model {request.model_id} trained successfully",
            data=result
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=ModelResponse)
async def predict_with_model(request: PredictRequest):
    """
    Generate predictions using a trained model.
    """
    try:
        model = model_factory.get_model(request.model_id)
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Generate predictions
        result = model.predict(df, **request.parameters)
        
        return ModelResponse(
            status="success",
            message=f"Predictions generated for model {request.model_id}",
            data=result
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/{model_id}")
async def evaluate_model(model_id: str, actual: List[float], predicted: List[float]):
    """
    Evaluate model performance.
    """
    try:
        model = model_factory.get_model(model_id)
        
        import numpy as np
        actual_array = np.array(actual)
        predicted_array = np.array(predicted)
        
        metrics = model.evaluate(actual_array, predicted_array)
        
        return ModelResponse(
            status="success",
            message=f"Model {model_id} evaluated successfully",
            data=metrics
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{model_id}")
async def get_model_status(model_id: str):
    """
    Get the current status of a model.
    """
    try:
        model = model_factory.get_model(model_id)
        info = model.get_info()
        
        return {
            "model_id": model_id,
            "status": info['status'],
            "accuracy": info.get('accuracy'),
            "last_run": info.get('last_run'),
            "parameters": getattr(model, 'parameters', {})
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset/{model_id}")
async def reset_model(model_id: str):
    """
    Reset a model to its initial state.
    """
    try:
        # Remove the model instance to force recreation
        if model_id in model_factory._model_instances:
            del model_factory._model_instances[model_id]
        
        # Create a fresh instance
        model = model_factory.create_model(model_id)
        
        return ModelResponse(
            status="success",
            message=f"Model {model_id} reset successfully"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_model_categories():
    """
    Get all available model categories.
    """
    try:
        models = model_factory.get_all_models()
        categories = set()
        
        for model in models:
            info = model.get_info()
            categories.add(info['category'])
        
        return {
            "categories": sorted(list(categories))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_model_types():
    """
    Get all available model types.
    """
    try:
        models = model_factory.get_all_models()
        types = set()
        
        for model in models:
            info = model.get_info()
            types.add(info['model_type'])
        
        return {
            "types": sorted(list(types))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))