'use client';

import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import {
  CpuChipIcon,
  ChartBarIcon,
  CpuChipIcon as BrainIcon,
  LightBulbIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

interface AIModel {
  id: string;
  name: string;
  type: 'prediction' | 'analysis' | 'strategy' | 'risk';
  description: string;
  accuracy?: number;
  status: 'active' | 'training' | 'inactive';
  lastUpdated: string;
  features: string[];
}

const availableModels: AIModel[] = [
  {
    id: 'lstm-predictor',
    name: 'LSTM Price Predictor',
    type: 'prediction',
    description: 'Long Short-Term Memory neural network for price prediction',
    accuracy: 78.5,
    status: 'active',
    lastUpdated: '2024-01-15',
    features: ['Price Forecasting', 'Trend Analysis', 'Volatility Prediction']
  },
  {
    id: 'random-forest',
    name: 'Random Forest Classifier',
    type: 'analysis',
    description: 'Ensemble method for market pattern recognition',
    accuracy: 82.3,
    status: 'active',
    lastUpdated: '2024-01-14',
    features: ['Pattern Recognition', 'Feature Importance', 'Risk Assessment']
  },
  {
    id: 'sentiment-analyzer',
    name: 'Sentiment Analysis Model',
    type: 'analysis',
    description: 'NLP model for market sentiment analysis',
    accuracy: 75.8,
    status: 'active',
    lastUpdated: '2024-01-13',
    features: ['News Analysis', 'Social Media Sentiment', 'Market Mood']
  },
  {
    id: 'risk-model',
    name: 'Risk Assessment Model',
    type: 'risk',
    description: 'Advanced risk modeling and portfolio optimization',
    accuracy: 85.2,
    status: 'training',
    lastUpdated: '2024-01-12',
    features: ['VaR Calculation', 'Portfolio Risk', 'Stress Testing']
  },
  {
    id: 'strategy-optimizer',
    name: 'Strategy Optimizer',
    type: 'strategy',
    description: 'Genetic algorithm for trading strategy optimization',
    accuracy: 73.9,
    status: 'active',
    lastUpdated: '2024-01-11',
    features: ['Strategy Backtesting', 'Parameter Optimization', 'Performance Analysis']
  },
  {
    id: 'elliott-wave',
    name: 'Elliott Wave Analyzer',
    type: 'analysis',
    description: 'Technical analysis using Elliott Wave theory',
    accuracy: 68.4,
    status: 'inactive',
    lastUpdated: '2024-01-10',
    features: ['Wave Pattern Detection', 'Fibonacci Levels', 'Trend Reversal']
  }
];

const ModelSelector: React.FC = () => {
  const { selectedModels, addSelectedModel, removeSelectedModel } = useStore();
  const [activeTab, setActiveTab] = useState('all');

  const getModelIcon = (type: string) => {
    switch (type) {
      case 'prediction':
        return <ChartBarIcon className="h-5 w-5" />;
      case 'analysis':
        return <BrainIcon className="h-5 w-5" />;
      case 'strategy':
        return <LightBulbIcon className="h-5 w-5" />;
      case 'risk':
        return <ExclamationTriangleIcon className="h-5 w-5" />;
      default:
        return <CpuChipIcon className="h-5 w-5" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
      case 'training':
        return <ClockIcon className="h-4 w-4 text-yellow-500" />;
      case 'inactive':
        return <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'training':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'inactive':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const filteredModels = activeTab === 'all' 
    ? availableModels 
    : availableModels.filter(model => model.type === activeTab);

  const isModelSelected = (modelId: string) => {
    return selectedModels.some(model => model.id === modelId);
  };

  const toggleModel = (model: AIModel) => {
    if (isModelSelected(model.id)) {
      removeSelectedModel(model.id);
    } else {
      // Convert AIModel to Model format expected by the store
      const storeModel = {
        id: model.id,
        name: model.name,
        category: 'cross-asset' as const, // Default category for AI models
        type: model.type === 'prediction' ? 'ml' as const : 
              model.type === 'analysis' ? 'technical' as const :
              model.type === 'strategy' ? 'ml' as const : 'ml' as const,
        description: model.description,
        accuracy: model.accuracy,
        lastRun: model.lastUpdated,
        isActive: model.status === 'active',
        status: model.status === 'active' ? 'idle' as const : 
                model.status === 'training' ? 'running' as const : 'idle' as const
      };
      addSelectedModel(storeModel);
    }
  };

  return (
    <div className="space-y-6">
      {/* Selected Models Summary */}
      {selectedModels.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Selected Models ({selectedModels.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {selectedModels.map((model) => (
                <Badge
                  key={model.id}
                  variant="secondary"
                  className="flex items-center space-x-2 px-3 py-1"
                >
                  {getModelIcon(model.type)}
                  <span>{model.name}</span>
                  <button
                    onClick={() => removeSelectedModel(model.id)}
                    className="ml-2 text-gray-500 hover:text-red-500"
                  >
                    ×
                  </button>
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Categories */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="prediction">Prediction</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="strategy">Strategy</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="space-y-4">
          {filteredModels.map((model) => (
            <Card
              key={model.id}
              className={`cursor-pointer transition-all hover:shadow-md ${
                isModelSelected(model.id)
                  ? 'ring-2 ring-primary-500 bg-primary-50 dark:bg-primary-900/10'
                  : ''
              }`}
              onClick={() => toggleModel(model)}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                      {getModelIcon(model.type)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {model.name}
                        </h3>
                        {getStatusIcon(model.status)}
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {model.description}
                      </p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                        {model.accuracy && (
                          <span>Accuracy: {model.accuracy}%</span>
                        )}
                        <span>Updated: {model.lastUpdated}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end space-y-2">
                    <Badge className={getStatusColor(model.status)}>
                      {model.status}
                    </Badge>
                    {isModelSelected(model.id) && (
                      <CheckCircleIcon className="h-5 w-5 text-primary-500" />
                    )}
                  </div>
                </div>
                
                {/* Features */}
                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex flex-wrap gap-1">
                    {model.features.map((feature, index) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className="text-xs"
                      >
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>
      </Tabs>

      {/* Action Buttons */}
      <div className="flex space-x-3">
        <Button
          variant="outline"
          onClick={() => {
            selectedModels.forEach(model => removeSelectedModel(model.id));
          }}
          disabled={selectedModels.length === 0}
        >
          Clear All
        </Button>
        <Button
          onClick={() => {
            availableModels
              .filter(model => model.status === 'active')
              .forEach(model => {
                if (!isModelSelected(model.id)) {
                  // Convert AIModel to Model format expected by the store
                  const storeModel = {
                    id: model.id,
                    name: model.name,
                    category: 'cross-asset' as const,
                    type: model.type === 'prediction' ? 'ml' as const : 
                          model.type === 'analysis' ? 'technical' as const :
                          model.type === 'strategy' ? 'ml' as const : 'ml' as const,
                    description: model.description,
                    accuracy: model.accuracy,
                    lastRun: model.lastUpdated,
                    isActive: model.status === 'active',
                    status: model.status === 'active' ? 'idle' as const : 
                            model.status === 'training' ? 'running' as const : 'idle' as const
                  };
                  addSelectedModel(storeModel);
                }
              });
          }}
        >
          Select All Active
        </Button>
      </div>
    </div>
  );
};

export default ModelSelector;