'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { marketDataService, CandlestickData, PriceData, TimeFrame } from '@/services/MarketDataService';
import { realtimeService } from '@/services/RealtimeService';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDownIcon,
  BarChart3Icon,
  CandlestickChartIcon,
  LineChartIcon,
  AreaChartIcon,
  VolumeXIcon,
  SettingsIcon,
  FullscreenIcon,
  ZoomInIcon,
  ZoomOutIcon
} from 'lucide-react';

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'buy' | 'sell' | 'neutral';
  color: string;
}

interface TradingViewChartProps {
  symbol: string;
  className?: string;
}

const TradingViewChart: React.FC<TradingViewChartProps> = ({ symbol, className = '' }) => {
  const [timeframe, setTimeframe] = useState<TimeFrame>('1d');
  const [chartType, setChartType] = useState('candlestick');
  const [indicators, setIndicators] = useState<string[]>(['RSI', 'MACD']);
  const [candleData, setCandleData] = useState<CandlestickData[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<TechnicalIndicator[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePercent, setPriceChangePercent] = useState(0);
  const chartRef = useRef<HTMLDivElement>(null);

  const timeframes: { value: TimeFrame; label: string }[] = [
    { value: '1m', label: '1M' },
    { value: '5m', label: '5M' },
    { value: '15m', label: '15M' },
    { value: '30m', label: '30M' },
    { value: '1h', label: '1H' },
    { value: '4h', label: '4H' },
    { value: '1d', label: '1D' },
    { value: '1w', label: '1W' },
    { value: '1M', label: '1M' }
  ];

  const chartTypes = [
    { value: 'candlestick', label: 'Candlestick', icon: CandlestickChartIcon },
    { value: 'line', label: 'Line', icon: LineChartIcon },
    { value: 'area', label: 'Area', icon: AreaChartIcon },
    { value: 'bar', label: 'Bar', icon: BarChart3Icon }
  ];

  const availableIndicators = [
    'RSI', 'MACD', 'Bollinger Bands', 'Moving Average', 'Stochastic', 
    'Williams %R', 'CCI', 'ADX', 'Parabolic SAR', 'Ichimoku'
  ];

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      try {
        // Load historical candlestick data
        const data = await marketDataService.getHistoricalData(symbol, timeframe);
        setCandleData(data);
        
        // Get current price
        const priceData = await marketDataService.getCurrentPrice(symbol);
        if (priceData) {
          setCurrentPrice(priceData.price);
          setPriceChange(priceData.change);
          setPriceChangePercent(priceData.changePercent);
        }
        
        // Get technical indicators
        const indicators = await marketDataService.getTechnicalIndicators(symbol);
        const rsiIndicator = indicators.find(ind => ind.name === 'RSI');
        const macdIndicator = indicators.find(ind => ind.name === 'MACD');
        const smaIndicator = indicators.find(ind => ind.name === 'SMA_20');
        
        setTechnicalIndicators([
          {
            name: 'RSI (14)',
            value: rsiIndicator?.value || 50,
            signal: rsiIndicator ? (rsiIndicator.value > 70 ? 'sell' : rsiIndicator.value < 30 ? 'buy' : 'neutral') : 'neutral',
            color: '#10b981'
          },
          {
            name: 'MACD',
            value: macdIndicator?.value || 0,
            signal: macdIndicator ? (macdIndicator.value > 0 ? 'buy' : 'sell') : 'neutral',
            color: '#3b82f6'
          },
          {
            name: 'Moving Average (20)',
            value: smaIndicator?.value || (priceData?.price || 0),
            signal: priceData && smaIndicator ? (priceData.price > smaIndicator.value ? 'buy' : 'sell') : 'neutral',
            color: '#f59e0b'
          }
        ]);
      } catch (error) {
        console.error('Error loading chart data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [symbol, timeframe]);

  // Real-time price updates
  useEffect(() => {
    const handlePriceUpdate = (data: PriceData) => {
      if (data.symbol === symbol) {
        setCurrentPrice(data.price);
        setPriceChange(data.change);
        setPriceChangePercent(data.changePercent);
      }
    };

    // Subscribe to real-time price updates
    realtimeService.on('priceUpdate', handlePriceUpdate);

    return () => {
      realtimeService.off('priceUpdate', handlePriceUpdate);
    };
  }, [symbol]);

  const renderCandlestickChart = () => {
    if (!candleData.length) return null;

    const maxPrice = Math.max(...candleData.map(d => d.high));
    const minPrice = Math.min(...candleData.map(d => d.low));
    const priceRange = maxPrice - minPrice;
    const chartHeight = 300;
    const chartWidth = 800;

    return (
      <div className="relative w-full h-80 bg-gray-900 rounded-lg overflow-hidden">
        <svg width="100%" height="100%" viewBox={`0 0 ${chartWidth} ${chartHeight + 50}`}>
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => (
            <line
              key={i}
              x1="0"
              y1={chartHeight * ratio}
              x2={chartWidth}
              y2={chartHeight * ratio}
              stroke="#374151"
              strokeWidth="0.5"
              opacity="0.3"
            />
          ))}

          {/* Candlesticks */}
          {candleData.slice(-100).map((candle, index) => {
            const x = (index / 100) * chartWidth;
            const candleWidth = chartWidth / 100 * 0.8;
            
            const openY = chartHeight - ((candle.open - minPrice) / priceRange) * chartHeight;
            const closeY = chartHeight - ((candle.close - minPrice) / priceRange) * chartHeight;
            const highY = chartHeight - ((candle.high - minPrice) / priceRange) * chartHeight;
            const lowY = chartHeight - ((candle.low - minPrice) / priceRange) * chartHeight;
            
            const isGreen = candle.close > candle.open;
            const color = isGreen ? '#10b981' : '#ef4444';
            
            return (
              <g key={index}>
                {/* High-Low line */}
                <line
                  x1={x + candleWidth / 2}
                  y1={highY}
                  x2={x + candleWidth / 2}
                  y2={lowY}
                  stroke={color}
                  strokeWidth="1"
                />
                
                {/* Open-Close body */}
                <rect
                  x={x}
                  y={Math.min(openY, closeY)}
                  width={candleWidth}
                  height={Math.abs(closeY - openY) || 1}
                  fill={isGreen ? color : 'transparent'}
                  stroke={color}
                  strokeWidth="1"
                />
              </g>
            );
          })}

          {/* Price labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio, i) => {
            const price = minPrice + (priceRange * (1 - ratio));
            return (
              <text
                key={i}
                x={chartWidth - 60}
                y={chartHeight * ratio + 5}
                fill="#9ca3af"
                fontSize="12"
                textAnchor="start"
              >
                ${price.toFixed(2)}
              </text>
            );
          })}
        </svg>

        {/* Volume chart at bottom */}
        <div className="absolute bottom-0 left-0 right-0 h-12 bg-gray-800">
          <svg width="100%" height="100%" viewBox={`0 0 ${chartWidth} 48`}>
            {candleData.slice(-100).map((candle, index) => {
              const x = (index / 100) * chartWidth;
              const barWidth = chartWidth / 100 * 0.8;
              const maxVolume = Math.max(...candleData.slice(-100).map(d => d.volume));
              const barHeight = (candle.volume / maxVolume) * 40;
              
              return (
                <rect
                  key={index}
                  x={x}
                  y={48 - barHeight}
                  width={barWidth}
                  height={barHeight}
                  fill={candle.close > candle.open ? '#10b981' : '#ef4444'}
                  opacity="0.6"
                />
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  return (
    <Card className={`w-full ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <CardTitle className="text-xl font-bold">{symbol}</CardTitle>
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold">
                ${currentPrice.toFixed(2)}
              </span>
              <Badge 
                variant={priceChange >= 0 ? "default" : "destructive"}
                className={`flex items-center space-x-1 ${
                  priceChange >= 0 ? 'bg-green-500' : 'bg-red-500'
                }`}
              >
                {priceChange >= 0 ? (
                  <TrendingUpIcon className="w-3 h-3" />
                ) : (
                  <TrendingDownIcon className="w-3 h-3" />
                )}
                <span>{priceChangePercent.toFixed(2)}%</span>
              </Badge>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <ZoomInIcon className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm">
              <ZoomOutIcon className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm">
              <FullscreenIcon className="w-4 h-4" />
            </Button>
            <Button variant="outline" size="sm">
              <SettingsIcon className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="flex items-center justify-between mt-4">
          {/* Timeframe selector */}
          <div className="flex items-center space-x-1">
            {timeframes.map((tf) => (
              <Button
                key={tf.value}
                variant={timeframe === tf.value ? "default" : "outline"}
                size="sm"
                onClick={() => setTimeframe(tf.value)}
                className="h-8 px-3"
              >
                {tf.label}
              </Button>
            ))}
          </div>

          {/* Chart type selector */}
          <div className="flex items-center space-x-1">
            {chartTypes.map((type) => {
              const IconComponent = type.icon;
              return (
                <Button
                  key={type.value}
                  variant={chartType === type.value ? "default" : "outline"}
                  size="sm"
                  onClick={() => setChartType(type.value)}
                  className="h-8 px-3"
                >
                  <IconComponent className="w-4 h-4" />
                </Button>
              );
            })}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center h-80 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-gray-600">Loading chart data...</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Main chart */}
            {renderCandlestickChart()}

            {/* Technical indicators */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {technicalIndicators.map((indicator, index) => (
                <div key={index} className="bg-gray-50 p-3 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      {indicator.name}
                    </span>
                    <Badge 
                      variant={
                        indicator.signal === 'buy' ? 'default' : 
                        indicator.signal === 'sell' ? 'destructive' : 'secondary'
                      }
                      className={`text-xs ${
                        indicator.signal === 'buy' ? 'bg-green-500' :
                        indicator.signal === 'sell' ? 'bg-red-500' : 'bg-gray-500'
                      }`}
                    >
                      {indicator.signal.toUpperCase()}
                    </Badge>
                  </div>
                  <div className="text-lg font-bold" style={{ color: indicator.color }}>
                    {indicator.value.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>

            {/* Indicator selector */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-700">Indicators:</span>
              <Select>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Add indicator" />
                </SelectTrigger>
                <SelectContent>
                  {availableIndicators.map((indicator) => (
                    <SelectItem key={indicator} value={indicator}>
                      {indicator}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TradingViewChart;