import { useState, useEffect, useCallback } from 'react';
import { useStore } from '../store/useStore';

export interface ModelPrediction {
  id: string;
  modelId: string;
  timestamp: string;
  prediction: number;
  confidence: number;
  signal?: 'buy' | 'sell' | 'hold';
  metadata?: Record<string, any>;
}

export interface ModelTrainingResult {
  status: 'success' | 'error';
  message?: string;
  accuracy?: number;
  metrics?: Record<string, number>;
}

export interface ModelPredictionResult {
  status: 'success' | 'error';
  message?: string;
  predictions?: any[];
  signals?: number[];
  confidence?: number[];
  timestamps?: string[];
  metadata?: Record<string, any>;
}

export const useModelPredictions = () => {
  const { 
    models, 
    modelPredictions, 
    addModelPrediction,
    updateModels,
    chartData 
  } = useStore();
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Helper function to update a single model
  const updateSingleModel = useCallback((updatedModel: Partial<any> & { id: string }) => {
    const updatedModels = models.map(model => 
      model.id === updatedModel.id ? { ...model, ...updatedModel } : model
    );
    updateModels(updatedModels);
  }, [models, updateModels]);

  // Fetch all available models from backend
  const fetchModels = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch('/api/models/');
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }
      
      const modelsData = await response.json();
      
      // Update store with fetched models
      modelsData.forEach((model: any) => {
        updateSingleModel({
          id: model.model_id,
          name: model.name,
          type: model.model_type,
          category: model.category,
          description: model.description,
          accuracy: model.accuracy || 0,
          timeframe: '1D', // Default timeframe
          status: model.status,
          lastRun: model.last_run,
          supportedAssets: ['all'] // Default to all assets
        });
      });
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models');
      console.error('Error fetching models:', err);
    } finally {
      setIsLoading(false);
    }
  }, [updateSingleModel]);

  // Train a specific model
  const trainModel = useCallback(async (
    modelId: string, 
    data?: any[], 
    parameters?: Record<string, any>
  ): Promise<ModelTrainingResult> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Use chart data if no data provided - get first available asset's data
      const availableAssetIds = Object.keys(chartData);
      const firstAssetData = availableAssetIds.length > 0 ? chartData[availableAssetIds[0]] : [];
      
      const trainingData = data || firstAssetData.map(point => ({
        timestamp: point.timestamp,
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
        volume: point.volume
      }));
      
      const response = await fetch('/api/models/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
          data: trainingData,
          parameters: parameters || {}
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to train model: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Update model status
      updateSingleModel({
        id: modelId,
        status: 'trained',
        accuracy: result.data?.accuracy || 0,
        lastRun: new Date().toISOString()
      });
      
      return {
        status: 'success',
        message: result.message,
        accuracy: result.data?.accuracy,
        metrics: result.data
      };
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to train model';
      setError(errorMessage);
      
      // Update model status to error
      updateSingleModel({
        id: modelId,
        status: 'error'
      });
      
      return {
        status: 'error',
        message: errorMessage
      };
    } finally {
      setIsLoading(false);
    }
  }, [chartData, updateSingleModel]);

  // Generate predictions with a model
  const predict = useCallback(async (
    modelId: string, 
    data?: any[], 
    parameters?: Record<string, any>
  ): Promise<ModelPredictionResult> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Use chart data if no data provided - get first available asset's data
      const availableAssetIds = Object.keys(chartData);
      const firstAssetData = availableAssetIds.length > 0 ? chartData[availableAssetIds[0]] : [];
      
      const predictionData = data || firstAssetData.map(point => ({
        timestamp: point.timestamp,
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
        volume: point.volume
      }));
      
      const response = await fetch('/api/models/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
          data: predictionData,
          parameters: parameters || {}
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to generate predictions: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.status === 'success' && result.data) {
        // Convert backend predictions to frontend format
        const predictions: ModelPrediction[] = [];
        
        if (result.data.signals && result.data.timestamps) {
          result.data.signals.forEach((signal: number, index: number) => {
            if (signal !== 0) { // Only include non-zero signals
              predictions.push({
                id: `${modelId}_${index}`,
                modelId,
                timestamp: result.data.timestamps[index],
                prediction: signal,
                confidence: result.data.confidence?.[index] || 0.5,
                signal: signal > 0 ? 'buy' : 'sell',
                metadata: {
                  index,
                  rawData: result.data
                }
              });
            }
          });
        }
        
        // Update predictions in store
        predictions.forEach(prediction => {
          const direction: 'bullish' | 'bearish' | 'neutral' = 
            prediction.signal === 'buy' ? 'bullish' : 
            prediction.signal === 'sell' ? 'bearish' : 'neutral';
            
          const storePrediction = {
            modelId: modelId,
            assetId: 'default', // Use a default asset ID since it's not provided
            prediction: {
              direction,
              confidence: prediction.confidence || 0.5,
              timeframe: '1d' as const,
              reasoning: `Model prediction with ${(prediction.confidence * 100).toFixed(1)}% confidence`
            },
            timestamp: prediction.timestamp
          };
          addModelPrediction(storePrediction);
        });
        
        // Update model status
        updateSingleModel({
          id: modelId,
          status: 'running',
          lastRun: new Date().toISOString()
        });
        
        return {
          status: 'success',
          message: result.message,
          predictions,
          signals: result.data.signals,
          confidence: result.data.confidence,
          timestamps: result.data.timestamps,
          metadata: result.data
        };
      }
      
      return {
        status: 'error',
        message: result.message || 'Failed to generate predictions'
      };
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate predictions';
      setError(errorMessage);
      
      return {
        status: 'error',
        message: errorMessage
      };
    } finally {
      setIsLoading(false);
    }
  }, [chartData, addModelPrediction, updateSingleModel]);

  // Get model status
  const getModelStatus = useCallback(async (modelId: string) => {
    try {
      const response = await fetch(`/api/models/status/${modelId}`);
      if (!response.ok) {
        throw new Error(`Failed to get model status: ${response.statusText}`);
      }
      
      const status = await response.json();
      
      // Update model in store
      updateSingleModel({
        id: modelId,
        status: status.status,
        accuracy: status.accuracy,
        lastRun: status.last_run
      });
      
      return status;
    } catch (err) {
      console.error('Error getting model status:', err);
      return null;
    }
  }, [updateSingleModel]);

  // Reset model
  const resetModel = useCallback(async (modelId: string) => {
    try {
      const response = await fetch(`/api/models/reset/${modelId}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`Failed to reset model: ${response.statusText}`);
      }
      
      // Update model status
      updateSingleModel({
        id: modelId,
        status: 'idle',
        accuracy: 0,
        lastRun: undefined
      });
      
      return true;
    } catch (err) {
      console.error('Error resetting model:', err);
      return false;
    }
  }, [updateSingleModel]);

  // Get models by category
  const getModelsByCategory = useCallback((category: string) => {
    return models.filter(model => 
      model.category.toLowerCase() === category.toLowerCase()
    );
  }, [models]);

  // Get predictions for a specific model
  const getPredictionsForModel = useCallback((modelId: string) => {
    return modelPredictions.filter(prediction => 
      prediction.modelId === modelId
    );
  }, [modelPredictions]);

  // Initialize models on mount
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  return {
    models,
    modelPredictions,
    isLoading,
    error,
    fetchModels,
    trainModel,
    predict,
    getModelStatus,
    resetModel,
    getModelsByCategory,
    getPredictionsForModel
  };
};