'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  ChartBarIcon,
  CogIcon,
  ArrowsPointingOutIcon,
  PlayIcon,
  PauseIcon,
  EyeIcon,
  EyeSlashIcon,
  CpuChipIcon as BrainIcon,
} from '@heroicons/react/24/outline';
import { useStore } from '../store/useStore';
import { useAssetsData, useModelPredictions } from '../hooks';
import { formatCurrency, formatPercentage, getChangeColor } from '../utils';
import LoadingSpinner from './ui/LoadingSpinner';
import FloatingAIAssistant from './FloatingAIAssistant';
import ChartOverlay from './ChartOverlay';
import { useRealTimeDataPipeline } from '../hooks/useRealTimeDataPipeline';

interface ChartData {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
  rsi?: number
  macd?: number
  bollinger_upper?: number
  bollinger_lower?: number
}

// Mock data generator
const generateMockData = (days: number = 30): ChartData[] => {
  const data: ChartData[] = []
  let basePrice = 50000
  
  for (let i = 0; i < days; i++) {
    const date = new Date()
    date.setDate(date.getDate() - (days - i))
    
    const volatility = 0.02
    const change = (Math.random() - 0.5) * volatility
    basePrice = basePrice * (1 + change)
    
    const open = basePrice
    const close = basePrice * (1 + (Math.random() - 0.5) * 0.01)
    const high = Math.max(open, close) * (1 + Math.random() * 0.005)
    const low = Math.min(open, close) * (1 - Math.random() * 0.005)
    
    data.push({
      timestamp: date.getTime(),
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume: Math.floor(Math.random() * 1000000),
      rsi: 30 + Math.random() * 40, // RSI between 30-70
      macd: (Math.random() - 0.5) * 100,
      bollinger_upper: close * 1.02,
      bollinger_lower: close * 0.98
    })
  }
  
  return data
}

const timeframes = [
  { label: '1D', value: '1d' },
  { label: '1W', value: '1w' },
  { label: '1M', value: '1m' },
  { label: '3M', value: '3m' },
  { label: '1Y', value: '1y' }
]

const chartTypes = [
  { label: 'Line', value: 'line' },
  { label: 'Candlestick', value: 'candlestick' },
  { label: 'Area', value: 'area' }
]

const TradingChart: React.FC = () => {
  const {
    selectedAssets,
    selectedModels,
    chartData,
    isLoading,
  } = useStore();
  
  const { subscribeToAssets } = useAssetsData();
  const modelPredictions = useModelPredictions();
  
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [isRealTime, setIsRealTime] = useState(true);
  const [showPredictions, setShowPredictions] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [timeframe, setTimeframe] = useState<'1m' | '5m' | '15m' | '1h' | '4h' | '1d'>('1h');
  const [showAIAssistant, setShowAIAssistant] = useState(false);
  const [showChartOverlay, setShowChartOverlay] = useState(false);
  
  // Real-time data pipeline integration
  const {
    isRunning: pipelineRunning,
    data: pipelineData,
    latestAnalysis: pipelineAnalysis,
    start: startPipeline,
    pause: pausePipeline,
    stop: stopPipeline
  } = useRealTimeDataPipeline();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null);
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Subscribe to real-time data for selected assets
  useEffect(() => {
    if (selectedAssets.length > 0) {
      const assetIds = selectedAssets.map(asset => asset.id);
      subscribeToAssets(assetIds);
    }
  }, [selectedAssets, subscribeToAssets]);

  // Subscribe to model predictions for selected models
  useEffect(() => {
    // This functionality would be handled by the WebSocket connection
    // when models are running and generating predictions
  }, [selectedModels, selectedAssets]);

  // Draw main price chart
  useEffect(() => {
    const currentChartData = currentAsset ? chartData[currentAsset.id] || [] : [];
    
    if (!canvasRef.current || !currentChartData || currentChartData.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Calculate price range
    const prices = currentChartData.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.1;

    // Drawing parameters
    const chartWidth = rect.width - 80;
    const chartHeight = rect.height - 60;
    const candleWidth = Math.max(2, chartWidth / currentChartData.length);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = (chartHeight / 5) * i + 20;
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(rect.width - 20, y);
      ctx.stroke();
    }

    if (chartType === 'candlestick') {
      // Draw candlesticks
      currentChartData.forEach((candle, index) => {
        const x = 40 + (index * candleWidth) + (candleWidth / 2);
        const openY = chartHeight - ((candle.open - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        const closeY = chartHeight - ((candle.close - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        const highY = chartHeight - ((candle.high - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        const lowY = chartHeight - ((candle.low - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;

        const isGreen = candle.close > candle.open;
        ctx.strokeStyle = isGreen ? '#10b981' : '#ef4444';
        ctx.fillStyle = isGreen ? '#10b981' : '#ef4444';
        ctx.lineWidth = 1;

        // Draw wick
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();

        // Draw body
        const bodyHeight = Math.abs(closeY - openY) || 1;
        const bodyY = Math.min(openY, closeY);
        const bodyWidth = Math.max(1, candleWidth * 0.6);
        
        if (isGreen) {
          ctx.fillRect(x - bodyWidth / 2, bodyY, bodyWidth, bodyHeight);
        } else {
          ctx.strokeRect(x - bodyWidth / 2, bodyY, bodyWidth, bodyHeight);
        }
      });
    } else if (chartType === 'line') {
      // Draw line chart
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      currentChartData.forEach((candle, index) => {
        const x = 40 + (index * candleWidth) + (candleWidth / 2);
        const y = chartHeight - ((candle.close - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    } else if (chartType === 'area') {
      // Draw area chart
      const gradient = ctx.createLinearGradient(0, 20, 0, chartHeight + 20);
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0.05)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      
      // Start from bottom left
      ctx.moveTo(40, chartHeight + 20);
      
      currentChartData.forEach((candle, index) => {
        const x = 40 + (index * candleWidth) + (candleWidth / 2);
        const y = chartHeight - ((candle.close - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        
        if (index === 0) {
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      // Close the path to bottom right
      const lastX = 40 + ((currentChartData.length - 1) * candleWidth) + (candleWidth / 2);
      ctx.lineTo(lastX, chartHeight + 20);
      ctx.closePath();
      ctx.fill();
      
      // Draw the line on top
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      currentChartData.forEach((candle, index) => {
        const x = 40 + (index * candleWidth) + (candleWidth / 2);
        const y = chartHeight - ((candle.close - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    }

    // Draw model predictions overlay
    if (showPredictions && modelPredictions && modelPredictions.length > 0) {
      modelPredictions.forEach((prediction, modelIndex) => {
        const colors = ['#f59e0b', '#8b5cf6', '#06b6d4', '#10b981', '#ef4444'];
        const color = colors[modelIndex % colors.length];
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        
        // Since prediction is a single value, create a simple prediction line
        const x = 40 + (currentChartData.length * candleWidth) + (candleWidth / 2);
        const predictionValue = Number(prediction.prediction.targetPrice || currentChartData[currentChartData.length - 1]?.close || 0);
        const y = chartHeight - ((predictionValue - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        
        // Connect to last actual price
        const lastX = 40 + ((currentChartData.length - 1) * candleWidth) + (candleWidth / 2);
        const lastY = chartHeight - ((currentChartData[currentChartData.length - 1].close - minPrice + padding) / (priceRange + 2 * padding)) * chartHeight + 20;
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // Draw price labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (priceRange * i / 5);
      const y = chartHeight - (i / 5) * chartHeight + 25;
      ctx.fillText(`$${price.toFixed(2)}`, 35, y);
    }

    // Draw time labels
    ctx.textAlign = 'center';
    const timeStep = Math.max(1, Math.floor(currentChartData.length / 6));
    for (let i = 0; i < currentChartData.length; i += timeStep) {
      const x = 40 + (i * candleWidth) + (candleWidth / 2);
      const time = new Date(currentChartData[i].timestamp);
      const timeStr = time.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      });
      ctx.fillText(timeStr, x, chartHeight + 40);
    }
  }, [selectedAssets, chartData, chartType, showPredictions, modelPredictions]);

  // Draw volume chart
  useEffect(() => {
    const currentChartData = currentAsset ? chartData[currentAsset.id] || [] : [];
    
    if (!volumeCanvasRef.current || !currentChartData || currentChartData.length === 0 || !showVolume) return;

    const canvas = volumeCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    ctx.clearRect(0, 0, rect.width, rect.height);

    const maxVolume = Math.max(...currentChartData.map(d => d.volume));
    const chartWidth = rect.width - 80;
    const chartHeight = rect.height - 20;
    const candleWidth = Math.max(2, chartWidth / currentChartData.length);

    currentChartData.forEach((candle, index) => {
      const x = 40 + (index * candleWidth);
      const height = (candle.volume / maxVolume) * chartHeight;
      const y = chartHeight - height;
      
      const isGreen = index > 0 ? candle.close > currentChartData[index - 1].close : true;
      ctx.fillStyle = isGreen ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)';
      ctx.fillRect(x, y, candleWidth * 0.8, height);
    });
  }, [selectedAssets, chartData, showVolume]);

  const timeframes = [
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '10m', label: '10m' },
    { value: '15m', label: '15m' },
    { value: '30m', label: '30m' },
    { value: '1h', label: '1h' },
    { value: '2h', label: '2h' },
    { value: '6h', label: '6h' },
    { value: '12h', label: '12h' },
    { value: '1d', label: '1d' },
    { value: '1w', label: '1w' },
    { value: '1M', label: '1M' },
    { value: '6M', label: '6M' },
    { value: '1y', label: '1y' },
    { value: '5y', label: '5y' },
    { value: 'all', label: 'All' }
  ];

  const chartTypes = [
    { value: 'candlestick', label: 'Candlestick' },
    { value: 'line', label: 'Line' },
    { value: 'area', label: 'Area' },
  ];

  const currentAsset = selectedAssets[0];
  const currentChartData = currentAsset ? chartData[currentAsset.id] || [] : [];
  const currentPrice = currentChartData && currentChartData.length > 0 ? currentChartData[currentChartData.length - 1] : null;

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 h-96">
        <div className="flex items-center justify-center h-full">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      {/* Chart Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Trading Chart
            </h3>
            {currentAsset && (
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500 dark:text-gray-400">Asset:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {currentAsset.symbol}
                  </span>
                </div>
                {currentPrice && (
                  <div className="flex items-center space-x-4">
                    <div className="text-lg font-semibold text-gray-900 dark:text-white">
                      {formatCurrency(currentPrice.close)}
                    </div>
                    <div className={`text-sm ${getChangeColor(currentPrice.close - currentPrice.open)}`}>
                      {formatPercentage((currentPrice.close - currentPrice.open) / currentPrice.open * 100)}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Timeframe Selector */}
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value as '1m' | '5m' | '15m' | '1h' | '4h' | '1d')}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {timeframes.map((tf) => (
                <option key={tf.value} value={tf.value}>
                  {tf.label}
                </option>
              ))}
            </select>
            
            {/* Chart Type Selector */}
            <select
              value={chartType}
              onChange={(e) => setChartType(e.target.value as any)}
              className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {chartTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
            
            {/* Toggle Buttons */}
            <button
              onClick={() => setShowPredictions(!showPredictions)}
              className={`flex items-center space-x-1 px-2 py-1 rounded text-xs transition-colors ${
                showPredictions
                  ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              {showPredictions ? <EyeIcon className="h-3 w-3" /> : <EyeSlashIcon className="h-3 w-3" />}
              <span>Predictions</span>
            </button>
            
            <button
              onClick={() => setShowVolume(!showVolume)}
              className={`flex items-center space-x-1 px-2 py-1 rounded text-xs transition-colors ${
                showVolume
                  ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              <ChartBarIcon className="h-3 w-3" />
              <span>Volume</span>
            </button>
            
            {/* Real-time Toggle */}
            <button
              onClick={() => setIsRealTime(!isRealTime)}
              className={`flex items-center space-x-1 px-3 py-1 rounded-lg text-sm transition-colors ${
                isRealTime
                  ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              {isRealTime ? (
                <PauseIcon className="h-4 w-4" />
              ) : (
                <PlayIcon className="h-4 w-4" />
              )}
              <span>{isRealTime ? 'Live' : 'Paused'}</span>
            </button>
            
            {/* AI Assistant */}
            <button 
              onClick={() => setShowAIAssistant(!showAIAssistant)}
              className={`flex items-center space-x-1 px-3 py-1 rounded-lg text-sm transition-colors ${
                showAIAssistant
                  ? 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              <BrainIcon className="h-4 w-4" />
              <span>AI</span>
            </button>
            
            {/* Chart Overlay Toggle */}
            <button 
              onClick={() => setShowChartOverlay(!showChartOverlay)}
              className={`flex items-center space-x-1 px-3 py-1 rounded-lg text-sm transition-colors ${
                showChartOverlay
                  ? 'bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-400'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              <ChartBarIcon className="h-4 w-4" />
              <span>Analysis</span>
            </button>
            
            {/* Pipeline Status Indicator */}
            {pipelineRunning && (
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 dark:bg-green-900/20 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-green-700 dark:text-green-300 font-medium">
                  Pipeline Active
                </span>
              </div>
            )}
            
            {/* Settings */}
            <button className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors">
              <CogIcon className="h-4 w-4" />
            </button>
            
            {/* Fullscreen */}
            <button className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors">
              <ArrowsPointingOutIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
      
      {/* Chart Canvas */}
      <div ref={chartContainerRef} className="relative">
        <canvas
          ref={canvasRef}
          className="w-full h-80"
          style={{ display: 'block' }}
        />
        
        {/* Volume Chart */}
        {showVolume && (
          <canvas
            ref={volumeCanvasRef}
            className="w-full h-20 border-t border-gray-200 dark:border-gray-700"
            style={{ display: 'block' }}
          />
        )}
        
        {/* Model Predictions Legend */}
        {showPredictions && modelPredictions.length > 0 && (
          <div className="absolute top-4 left-4 bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-200 dark:border-gray-700">
            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">Model Predictions</h4>
            <div className="space-y-1">
              {modelPredictions.map((prediction, index) => {
                const colors = ['#f59e0b', '#8b5cf6', '#06b6d4', '#10b981', '#ef4444'];
                const color = colors[index % colors.length];
                
                return (
                  <div key={prediction.modelId} className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-0.5" 
                      style={{ backgroundColor: color, borderStyle: 'dashed' }}
                    ></div>
                    <span className="text-xs text-gray-700 dark:text-gray-300">
                      {selectedModels.find(m => m.id === prediction.modelId)?.name || 'Unknown Model'}
                    </span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {(prediction.prediction.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        
        {/* Chart Overlay */}
        {showChartOverlay && currentAsset && (
          <ChartOverlay
            symbol={currentAsset.symbol}
            isVisible={showChartOverlay}
            onToggleVisibility={() => setShowChartOverlay(false)}
            position="top-right"
          />
        )}
        
        {/* No Data Message */}
        {(!currentChartData || currentChartData.length === 0) && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500 dark:text-gray-400">
                {selectedAssets.length === 0 
                  ? 'Select an asset to view chart data'
                  : 'Loading chart data...'
                }
              </p>
            </div>
          </div>
        )}
      </div>

      
      {/* Chart Info */}
      {currentPrice && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
            <div>
              <span className="text-gray-500 dark:text-gray-400">Open:</span>
              <span className="ml-2 font-medium text-gray-900 dark:text-white">
                {formatCurrency(currentPrice.open)}
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">High:</span>
              <span className="ml-2 font-medium text-gray-900 dark:text-white">
                {formatCurrency(currentPrice.high)}
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Low:</span>
              <span className="ml-2 font-medium text-gray-900 dark:text-white">
                {formatCurrency(currentPrice.low)}
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Close:</span>
              <span className="ml-2 font-medium text-gray-900 dark:text-white">
                {formatCurrency(currentPrice.close)}
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Volume:</span>
              <span className="ml-2 font-medium text-gray-900 dark:text-white">
                {currentPrice.volume.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}
      
      {/* Floating AI Assistant */}
      {showAIAssistant && (
        <FloatingAIAssistant
          symbol={currentAsset?.symbol || 'UNKNOWN'}
          onClose={() => setShowAIAssistant(false)}
          chartContainerRef={chartContainerRef}
          position={{ x: 20, y: 20 }}
        />
      )}
    </div>
  );
};

export default TradingChart;