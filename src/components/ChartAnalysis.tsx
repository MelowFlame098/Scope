'use client';

import React, { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import {
  PhotoIcon,
  ChartBarIcon,
  CpuChipIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  MinusIcon,
  EyeIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import { chartAnalysisApi } from '../lib/api';

interface ChartValidation {
  is_valid: boolean;
  chart_type: string;
  quality_score: number;
  issues: string[];
}

interface ExtractedFeatures {
  ohlc_count: number;
  indicators_count: number;
  custom_indicators_count: number;
  confidence: number;
}

interface Forecast {
  direction: string;
  confidence: number;
}

interface StrategyRecommendation {
  primary_recommendation: string;
  confidence: number;
  strategies_count: number;
}

interface ExecutionSignal {
  action: string;
  confidence: number;
  signal_strength: string;
  position_size: number;
  stop_loss?: number;
  take_profit?: number;
  reasoning: string;
}

interface ChartAnalysisResult {
  success: boolean;
  symbol: string;
  timeframe: string;
  processing_time: number;
  chart_validation?: ChartValidation;
  extracted_features?: ExtractedFeatures;
  forecasts?: {
    price_forecast?: Forecast;
    volatility_forecast?: Forecast;
  };
  strategy_recommendations?: StrategyRecommendation;
  execution_signals?: ExecutionSignal[];
  error_message?: string;
  metadata?: any;
}

const ChartAnalysis: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1D');
  const [customIndicators, setCustomIndicators] = useState('');
  const [enableForecasting, setEnableForecasting] = useState(true);
  const [enableExecution, setEnableExecution] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<ChartAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }

      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }

      setSelectedFile(file);
      setError(null);

      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  }, []);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      const fakeEvent = {
        target: { files: [file] }
      } as unknown as React.ChangeEvent<HTMLInputElement>;
      handleFileSelect(fakeEvent);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  }, []);

  const analyzeChart = async () => {
    if (!selectedFile) {
      setError('Please select a chart image first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('chart_file', selectedFile);
      formData.append('symbol', symbol.toUpperCase());
      formData.append('timeframe', timeframe);
      formData.append('custom_indicators', customIndicators);
      formData.append('enable_forecasting', enableForecasting.toString());
      formData.append('enable_execution', enableExecution.toString());

      const response = await chartAnalysisApi.analyzeChart(formData);
      
      if (response.data) {
        setAnalysisResult(response.data);
      } else {
        setError(response.error || 'Analysis failed');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to analyze chart');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearAnalysis = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getActionIcon = (action: string) => {
    switch (action.toLowerCase()) {
      case 'buy':
        return <ArrowUpIcon className="h-4 w-4 text-green-500" />;
      case 'sell':
        return <ArrowDownIcon className="h-4 w-4 text-red-500" />;
      default:
        return <MinusIcon className="h-4 w-4 text-gray-500" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-500';
    if (confidence >= 0.6) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getSignalStrengthColor = (strength: string) => {
    switch (strength.toLowerCase()) {
      case 'strong':
        return 'bg-green-500';
      case 'moderate':
        return 'bg-yellow-500';
      case 'weak':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <PhotoIcon className="h-5 w-5" />
            Chart Upload & Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* File Upload */}
          <div
            className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition-colors"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
            {previewUrl ? (
              <div className="space-y-4">
                <img
                  src={previewUrl}
                  alt="Chart preview"
                  className="max-h-48 mx-auto rounded-lg shadow-md"
                />
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {selectedFile?.name} ({(selectedFile?.size || 0 / 1024 / 1024).toFixed(2)} MB)
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <PhotoIcon className="h-12 w-12 mx-auto text-gray-400" />
                <p className="text-gray-600 dark:text-gray-400">
                  Drop your chart image here or click to browse
                </p>
                <p className="text-xs text-gray-500">
                  Supports PNG, JPEG, GIF, BMP (max 10MB)
                </p>
              </div>
            )}
          </div>

          {/* Analysis Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="symbol">Symbol</Label>
              <Input
                id="symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="e.g., AAPL, TSLA"
              />
            </div>
            <div>
              <Label htmlFor="timeframe">Timeframe</Label>
              <select
                id="timeframe"
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="30m">30 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="4h">4 Hours</option>
                <option value="1D">1 Day</option>
                <option value="1W">1 Week</option>
                <option value="1M">1 Month</option>
              </select>
            </div>
            <div className="md:col-span-2">
              <Label htmlFor="custom-indicators">Custom Indicators (comma-separated)</Label>
              <Input
                id="custom-indicators"
                value={customIndicators}
                onChange={(e) => setCustomIndicators(e.target.value)}
                placeholder="e.g., MyRSI, CustomMA, TrendIndicator"
              />
            </div>
          </div>

          {/* Options */}
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={enableForecasting}
                onChange={(e) => setEnableForecasting(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">Enable Forecasting</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={enableExecution}
                onChange={(e) => setEnableExecution(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">Enable Execution Signals</span>
            </label>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={analyzeChart}
              disabled={!selectedFile || isAnalyzing}
              className="flex items-center gap-2"
            >
              {isAnalyzing ? (
                <ClockIcon className="h-4 w-4 animate-spin" />
              ) : (
                <CpuChipIcon className="h-4 w-4" />
              )}
              {isAnalyzing ? 'Analyzing...' : 'Analyze Chart'}
            </Button>
            <Button variant="outline" onClick={clearAnalysis}>
              Clear
            </Button>
          </div>

          {/* Error Display */}
          {error && (
            <Alert className="border-red-200 bg-red-50 dark:bg-red-900/20">
              <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
              <AlertDescription className="text-red-700 dark:text-red-400">
                {error}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysisResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ChartBarIcon className="h-5 w-5" />
              Analysis Results
              {analysisResult.success ? (
                <CheckCircleIcon className="h-5 w-5 text-green-500" />
              ) : (
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analysisResult.success ? (
              <Tabs defaultValue="overview" className="space-y-4">
                <TabsList>
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="validation">Chart Validation</TabsTrigger>
                  <TabsTrigger value="features">Extracted Features</TabsTrigger>
                  <TabsTrigger value="forecasts">Forecasts</TabsTrigger>
                  <TabsTrigger value="strategies">Strategies</TabsTrigger>
                  <TabsTrigger value="signals">Execution Signals</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Symbol</div>
                      <div className="text-lg font-semibold">{analysisResult.symbol}</div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Timeframe</div>
                      <div className="text-lg font-semibold">{analysisResult.timeframe}</div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                      <div className="text-sm text-gray-600 dark:text-gray-400">Processing Time</div>
                      <div className="text-lg font-semibold">{analysisResult.processing_time.toFixed(2)}s</div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="validation" className="space-y-4">
                  {analysisResult.chart_validation && (
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        {analysisResult.chart_validation.is_valid ? (
                          <CheckCircleIcon className="h-5 w-5 text-green-500" />
                        ) : (
                          <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                        )}
                        <span className="font-medium">
                          Chart {analysisResult.chart_validation.is_valid ? 'Valid' : 'Invalid'}
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label>Chart Type</Label>
                          <Badge variant="outline">
                            {analysisResult.chart_validation.chart_type || 'Unknown'}
                          </Badge>
                        </div>
                        <div>
                          <Label>Quality Score</Label>
                          <div className="flex items-center gap-2">
                            <Progress value={analysisResult.chart_validation.quality_score * 100} className="flex-1" />
                            <span className="text-sm font-medium">
                              {(analysisResult.chart_validation.quality_score * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                      {analysisResult.chart_validation.issues.length > 0 && (
                        <div>
                          <Label>Issues</Label>
                          <ul className="list-disc list-inside space-y-1 text-sm text-gray-600 dark:text-gray-400">
                            {analysisResult.chart_validation.issues.map((issue, index) => (
                              <li key={index}>{issue}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="features" className="space-y-4">
                  {analysisResult.extracted_features && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                        <div className="text-sm text-blue-600 dark:text-blue-400">OHLC Data Points</div>
                        <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                          {analysisResult.extracted_features.ohlc_count}
                        </div>
                      </div>
                      <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                        <div className="text-sm text-green-600 dark:text-green-400">Standard Indicators</div>
                        <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                          {analysisResult.extracted_features.indicators_count}
                        </div>
                      </div>
                      <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                        <div className="text-sm text-purple-600 dark:text-purple-400">Custom Indicators</div>
                        <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                          {analysisResult.extracted_features.custom_indicators_count}
                        </div>
                      </div>
                      <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                        <div className="text-sm text-orange-600 dark:text-orange-400">Confidence</div>
                        <div className={`text-2xl font-bold ${getConfidenceColor(analysisResult.extracted_features.confidence)}`}>
                          {(analysisResult.extracted_features.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="forecasts" className="space-y-4">
                  {analysisResult.forecasts && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {analysisResult.forecasts.price_forecast && (
                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                          <h4 className="font-medium mb-2">Price Forecast</h4>
                          <div className="flex items-center gap-2">
                            {getActionIcon(analysisResult.forecasts.price_forecast.direction)}
                            <span className="font-medium">
                              {analysisResult.forecasts.price_forecast.direction}
                            </span>
                            <Badge className={getConfidenceColor(analysisResult.forecasts.price_forecast.confidence)}>
                              {(analysisResult.forecasts.price_forecast.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                        </div>
                      )}
                      {analysisResult.forecasts.volatility_forecast && (
                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                          <h4 className="font-medium mb-2">Volatility Forecast</h4>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">
                              {analysisResult.forecasts.volatility_forecast.direction}
                            </span>
                            <Badge className={getConfidenceColor(analysisResult.forecasts.volatility_forecast.confidence)}>
                              {(analysisResult.forecasts.volatility_forecast.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="strategies" className="space-y-4">
                  {analysisResult.strategy_recommendations && (
                    <div className="space-y-4">
                      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                        <h4 className="font-medium mb-2">Primary Recommendation</h4>
                        <div className="flex items-center gap-2">
                          <LightBulbIcon className="h-5 w-5 text-yellow-500" />
                          <span className="font-medium">
                            {analysisResult.strategy_recommendations.primary_recommendation}
                          </span>
                          <Badge className={getConfidenceColor(analysisResult.strategy_recommendations.confidence)}>
                            {(analysisResult.strategy_recommendations.confidence * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {analysisResult.strategy_recommendations.strategies_count} strategies analyzed
                      </div>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="signals" className="space-y-4">
                  {analysisResult.execution_signals && analysisResult.execution_signals.length > 0 ? (
                    <div className="space-y-4">
                      {analysisResult.execution_signals.map((signal, index) => (
                        <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                              {getActionIcon(signal.action)}
                              <span className="font-medium text-lg">{signal.action.toUpperCase()}</span>
                              <Badge className={`${getSignalStrengthColor(signal.signal_strength)} text-white`}>
                                {signal.signal_strength}
                              </Badge>
                            </div>
                            <Badge className={getConfidenceColor(signal.confidence)}>
                              {(signal.confidence * 100).toFixed(1)}% confidence
                            </Badge>
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                            <div>
                              <Label>Position Size</Label>
                              <div className="font-medium">{(signal.position_size * 100).toFixed(1)}%</div>
                            </div>
                            {signal.stop_loss && (
                              <div>
                                <Label>Stop Loss</Label>
                                <div className="font-medium text-red-600">${signal.stop_loss.toFixed(2)}</div>
                              </div>
                            )}
                            {signal.take_profit && (
                              <div>
                                <Label>Take Profit</Label>
                                <div className="font-medium text-green-600">${signal.take_profit.toFixed(2)}</div>
                              </div>
                            )}
                          </div>
                          <div>
                            <Label>Reasoning</Label>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              {signal.reasoning}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 py-8">
                      <CpuChipIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>No execution signals generated</p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            ) : (
              <Alert className="border-red-200 bg-red-50 dark:bg-red-900/20">
                <ExclamationTriangleIcon className="h-4 w-4 text-red-500" />
                <AlertDescription className="text-red-700 dark:text-red-400">
                  Analysis failed: {analysisResult.error_message}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ChartAnalysis;