"use client";

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  ChartBarIcon,
  Cog6ToothIcon,
  ArrowsPointingOutIcon,
  PlusIcon
} from '@heroicons/react/24/outline';

interface CandlestickData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TradingChartProps {
  symbol: string;
  height?: number;
}

export const TradingChart: React.FC<TradingChartProps> = ({ 
  symbol, 
  height = 400 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [timeframe, setTimeframe] = useState('1D');
  const [chartType, setChartType] = useState('candlestick');
  const [indicators, setIndicators] = useState<string[]>(['MA20', 'Volume']);
  const [data, setData] = useState<CandlestickData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Mock data generation
  useEffect(() => {
    const generateMockData = () => {
      const mockData: CandlestickData[] = [];
      let basePrice = 155.50;
      const now = Date.now();
      
      for (let i = 100; i >= 0; i--) {
        const timestamp = now - (i * 24 * 60 * 60 * 1000); // Daily data
        const open = basePrice + (Math.random() - 0.5) * 2;
        const volatility = Math.random() * 3;
        const high = open + Math.random() * volatility;
        const low = open - Math.random() * volatility;
        const close = low + Math.random() * (high - low);
        const volume = Math.floor(Math.random() * 1000000) + 100000;
        
        mockData.push({
          timestamp,
          open,
          high,
          low,
          close,
          volume
        });
        
        basePrice = close + (Math.random() - 0.5) * 0.5;
      }
      
      setData(mockData);
      setIsLoading(false);
    };

    generateMockData();
  }, [symbol, timeframe]);

  // Canvas drawing logic
  useEffect(() => {
    if (!canvasRef.current || data.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear canvas
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Calculate price range
    const prices = data.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Chart dimensions
    const padding = 40;
    const chartWidth = rect.width - padding * 2;
    const chartHeight = rect.height - padding * 2;
    const candleWidth = chartWidth / data.length * 0.8;

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(rect.width - padding, y);
      ctx.stroke();
      
      // Price labels
      const price = maxPrice - (priceRange / 5) * i;
      ctx.fillStyle = '#9ca3af';
      ctx.font = '12px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(price.toFixed(2), padding - 5, y + 4);
    }

    // Vertical grid lines
    const timeStep = Math.floor(data.length / 6);
    for (let i = 0; i < data.length; i += timeStep) {
      const x = padding + (chartWidth / data.length) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, rect.height - padding);
      ctx.stroke();
      
      // Time labels
      if (data[i]) {
        const date = new Date(data[i].timestamp);
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(
          date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          x,
          rect.height - padding + 20
        );
      }
    }

    // Draw candlesticks
    data.forEach((candle, index) => {
      const x = padding + (chartWidth / data.length) * index + (chartWidth / data.length) * 0.1;
      const centerX = x + candleWidth / 2;
      
      // Calculate y positions
      const highY = padding + ((maxPrice - candle.high) / priceRange) * chartHeight;
      const lowY = padding + ((maxPrice - candle.low) / priceRange) * chartHeight;
      const openY = padding + ((maxPrice - candle.open) / priceRange) * chartHeight;
      const closeY = padding + ((maxPrice - candle.close) / priceRange) * chartHeight;
      
      const isGreen = candle.close > candle.open;
      const color = isGreen ? '#10b981' : '#ef4444';
      
      // Draw wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(centerX, highY);
      ctx.lineTo(centerX, lowY);
      ctx.stroke();
      
      // Draw body
      ctx.fillStyle = color;
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY);
      ctx.fillRect(x, bodyTop, candleWidth, Math.max(bodyHeight, 1));
    });

    // Draw moving average if enabled
    if (indicators.includes('MA20')) {
      const ma20Data = calculateMovingAverage(data, 20);
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      ma20Data.forEach((ma, index) => {
        if (ma !== null) {
          const x = padding + (chartWidth / data.length) * index + (chartWidth / data.length) * 0.5;
          const y = padding + ((maxPrice - ma) / priceRange) * chartHeight;
          
          if (index === 0 || ma20Data[index - 1] === null) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
      });
      ctx.stroke();
    }

  }, [data, indicators, chartType]);

  const calculateMovingAverage = (data: CandlestickData[], period: number): (number | null)[] => {
    return data.map((_, index) => {
      if (index < period - 1) return null;
      
      const sum = data.slice(index - period + 1, index + 1)
        .reduce((acc, candle) => acc + candle.close, 0);
      return sum / period;
    });
  };

  const timeframes = [
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '1h', label: '1h' },
    { value: '4h', label: '4h' },
    { value: '1D', label: '1D' },
    { value: '1W', label: '1W' }
  ];

  const availableIndicators = [
    { value: 'MA20', label: 'MA(20)' },
    { value: 'MA50', label: 'MA(50)' },
    { value: 'RSI', label: 'RSI' },
    { value: 'MACD', label: 'MACD' },
    { value: 'Volume', label: 'Volume' },
    { value: 'Bollinger', label: 'Bollinger Bands' }
  ];

  const toggleIndicator = (indicator: string) => {
    setIndicators(prev => 
      prev.includes(indicator)
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator]
    );
  };

  return (
    <Card className="bg-gray-800 border-gray-700 h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white text-lg flex items-center">
            <ChartBarIcon className="w-5 h-5 mr-2" />
            {symbol} Chart
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="text-gray-300">
              {isLoading ? 'Loading...' : 'Live'}
            </Badge>
            <Button size="sm" variant="outline">
              <ArrowsPointingOutIcon className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        {/* Chart Controls */}
        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center space-x-2">
            {/* Timeframe Selector */}
            <div className="flex space-x-1">
              {timeframes.map((tf) => (
                <Button
                  key={tf.value}
                  size="sm"
                  variant={timeframe === tf.value ? "default" : "outline"}
                  onClick={() => setTimeframe(tf.value)}
                  className="px-2 py-1 text-xs"
                >
                  {tf.label}
                </Button>
              ))}
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Chart Type */}
            <Select value={chartType} onValueChange={setChartType}>
              <SelectTrigger className="w-32 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="candlestick">Candlestick</SelectItem>
                <SelectItem value="line">Line</SelectItem>
                <SelectItem value="area">Area</SelectItem>
              </SelectContent>
            </Select>
            
            {/* Indicators */}
            <Button size="sm" variant="outline">
              <PlusIcon className="w-4 h-4 mr-1" />
              Indicators
            </Button>
            
            <Button size="sm" variant="outline">
              <Cog6ToothIcon className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        {/* Active Indicators */}
        {indicators.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {indicators.map((indicator) => (
              <Badge
                key={indicator}
                variant="secondary"
                className="cursor-pointer hover:bg-red-600"
                onClick={() => toggleIndicator(indicator)}
              >
                {indicator} ×
              </Badge>
            ))}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="p-0">
        <div className="relative" style={{ height: `${height}px` }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ display: 'block' }}
          />
          
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800/50">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin mx-auto mb-2"></div>
                <p className="text-gray-400 text-sm">Loading chart data...</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};