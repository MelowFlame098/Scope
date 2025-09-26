'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  ArrowTrendingUpIcon as TrendingUpIcon,
  ArrowTrendingDownIcon as TrendingDownIcon,
  ChartBarIcon as BarChart3Icon,
  PresentationChartLineIcon as LineChartIcon,
  ChartPieIcon as PieChartIcon,
  PresentationChartBarIcon as CandlestickChartIcon,
  PresentationChartLineIcon as AreaChartIcon,
  ChartBarIcon as ScatterChartIcon,
  Cog6ToothIcon as Settings2Icon,
  Cog6ToothIcon as SettingsIcon,
  MagnifyingGlassPlusIcon as ZoomInIcon,
  MagnifyingGlassMinusIcon as ZoomOutIcon,
  ArrowDownTrayIcon as DownloadIcon,
  ArrowsPointingOutIcon as FullscreenIcon,
  ArrowPathIcon as RefreshCwIcon,
  ExclamationTriangleIcon as AlertTriangleIcon,
  InformationCircleIcon as InfoIcon,
  PlayIcon,
  PauseIcon,
  BackwardIcon as SkipBackIcon,
  ForwardIcon as SkipForwardIcon,
  ArrowsPointingOutIcon as MaximizeIcon,
  ArrowsPointingInIcon as MinimizeIcon
} from '@heroicons/react/24/outline';
import TechnicalsInterface from './TechnicalsInterface';
// UI Components - using inline styles for now
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 pb-4 ${className}`}>{children}</div>
);

const CardTitle = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <h3 className={`text-lg font-semibold text-gray-900 dark:text-white ${className}`}>{children}</h3>
);

const CardContent = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 pt-0 ${className}`}>{children}</div>
);

const Button = ({ children, onClick, className = '', variant = 'default', disabled = false, size = 'default' }: {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  variant?: 'default' | 'outline' | 'ghost';
  disabled?: boolean;
  size?: 'default' | 'sm' | 'lg';
}) => {
  const baseClasses = 'rounded-lg font-medium transition-colors';
  const sizeClasses = size === 'sm' ? 'px-3 py-1.5 text-sm' : size === 'lg' ? 'px-6 py-3 text-lg' : 'px-4 py-2';
  const variantClasses = variant === 'outline'
    ? 'border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
    : variant === 'ghost'
    ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
    : 'bg-blue-600 text-white hover:bg-blue-700';
  const disabledClasses = disabled ? 'opacity-50 cursor-not-allowed' : '';
  
  return (
    <button 
      onClick={disabled ? undefined : onClick} 
      disabled={disabled}
      className={`${baseClasses} ${sizeClasses} ${variantClasses} ${disabledClasses} ${className}`}
    >
      {children}
    </button>
  );
};

const Input = ({ value, onChange, placeholder, className = '', type = 'text', min, max }: {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
  className?: string;
  type?: string;
  min?: string;
  max?: string;
}) => (
  <input
    type={type}
    min={min}
    max={max}
    value={value}
    onChange={onChange}
    placeholder={placeholder}
    className={`px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${className}`}
  />
);

const Label = ({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) => (
  <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
    {children}
  </label>
);
// Additional UI Components
const Select = ({ value, onValueChange, children }: {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}) => (
  <select
    value={value}
    onChange={(e) => onValueChange(e.target.value)}
    className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
  >
    {children}
  </select>
);

const SelectTrigger = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={className}>{children}</div>
);
const SelectValue = ({ placeholder }: { placeholder?: string }) => <option value="">{placeholder}</option>;
const SelectContent = ({ children }: { children: React.ReactNode }) => <>{children}</>;
const SelectItem = ({ value, children }: { value: string; children: React.ReactNode }) => (
  <option value={value}>{children}</option>
);

const Tabs = ({ value, onValueChange, children }: {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}) => (
  <div className="w-full">{children}</div>
);

const TabsList = ({ children }: { children: React.ReactNode }) => (
  <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg mb-4">
    {children}
  </div>
);

const TabsTrigger = ({ value, children, isActive, onClick }: {
  value: string;
  children: React.ReactNode;
  isActive?: boolean;
  onClick?: () => void;
}) => (
  <button
    onClick={onClick}
    className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
    }`}
  >
    {children}
  </button>
);

const TabsContent = ({ value, children, isActive }: {
  value: string;
  children: React.ReactNode;
  isActive?: boolean;
}) => (
  <div className={isActive ? 'block' : 'hidden'}>
    {children}
  </div>
);
// Final UI Components
const Badge = ({ children, variant = 'default', className = '' }: {
  children: React.ReactNode;
  variant?: 'default' | 'secondary' | 'destructive';
  className?: string;
}) => {
  const variantClasses = {
    default: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    secondary: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
    destructive: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
  };
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variantClasses[variant]} ${className}`}>
      {children}
    </span>
  );
};

const Alert = ({ children, className = '', variant }: { children: React.ReactNode; className?: string; variant?: string }) => (
  <div className={`p-4 rounded-lg border ${variant === 'destructive' ? 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20' : 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20'} ${className}`}>
    {children}
  </div>
);

const AlertDescription = ({ children }: { children: React.ReactNode }) => (
  <div className="text-sm text-blue-700 dark:text-blue-300">{children}</div>
);


interface ChartData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  indicators?: Record<string, number>;
}

interface TechnicalIndicator {
  name: string;
  type: 'SMA' | 'EMA' | 'RSI' | 'MACD' | 'BOLLINGER_BANDS' | 'STOCHASTIC' | 'WILLIAMS_R' | 'CCI' | 'ADX';
  period: number;
  parameters?: Record<string, any>;
  visible: boolean;
  color: string;
}

interface Pattern {
  type: string;
  confidence: number;
  startTime: string;
  endTime: string;
  description: string;
  bullish: boolean;
}

interface ChartSettings {
  chartType: 'CANDLESTICK' | 'LINE' | 'BAR' | 'AREA' | 'HEIKIN_ASHI';
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';
  showVolume: boolean;
  showGrid: boolean;
  theme: 'light' | 'dark';
  autoRefresh: boolean;
  refreshInterval: number;
}

// Helper function to determine asset type from symbol
const getAssetType = (symbol: string): string => {
  // Crypto patterns
  if (symbol.includes('BTC') || symbol.includes('ETH') || symbol.includes('USDT') || 
      symbol.includes('BNB') || symbol.includes('ADA') || symbol.includes('SOL') ||
      symbol.includes('DOGE') || symbol.includes('MATIC') || symbol.includes('AVAX')) {
    return 'crypto';
  }
  
  // Forex patterns (currency pairs)
  if (symbol.length === 6 && /^[A-Z]{6}$/.test(symbol)) {
    const currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD'];
    const base = symbol.substring(0, 3);
    const quote = symbol.substring(3, 6);
    if (currencies.includes(base) && currencies.includes(quote)) {
      return 'forex';
    }
  }
  
  // Futures patterns
  if (symbol.includes('F') && (symbol.includes('CL') || symbol.includes('GC') || 
      symbol.includes('ES') || symbol.includes('NQ') || symbol.includes('YM'))) {
    return 'futures';
  }
  
  // Index patterns
  if (symbol.includes('SPX') || symbol.includes('NDX') || symbol.includes('DJI') ||
      symbol.includes('VIX') || symbol.includes('RUT') || symbol.includes('FTSE') ||
      symbol.includes('DAX') || symbol.includes('NIKKEI')) {
    return 'index';
  }
  
  // Default to stock for everything else
  return 'stock';
};

// Helper function to get indicator color
const getIndicatorColor = (indicatorName: string): string => {
  const colors = {
    'RSI': '#ff6b6b',
    'MACD': '#4ecdc4',
    'SMA': '#45b7d1',
    'EMA': '#96ceb4',
    'BOLLINGER_BANDS': '#feca57',
    'STOCHASTIC': '#ff9ff3',
    'WILLIAMS_R': '#54a0ff',
    'CCI': '#5f27cd',
    'ADX': '#00d2d3',
    'ARIMA': '#ff6348',
    'GARCH': '#2ed573',
    'LSTM': '#3742fa',
    'XGBOOST': '#ff4757'
  };
  
  return colors[indicatorName as keyof typeof colors] || '#6c5ce7';
};

const AdvancedCharting: React.FC = () => {
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([]);
  const [patterns, setPatterns] = useState<Pattern[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [enabledIndicators, setEnabledIndicators] = useState<string[]>([]);
  const [selectedAssetType, setSelectedAssetType] = useState<'crypto' | 'stock' | 'forex' | 'futures' | 'index'>('stock');
  const chartRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [settings, setSettings] = useState<ChartSettings>({
    chartType: 'CANDLESTICK',
    timeframe: '1h',
    showVolume: true,
    showGrid: true,
    theme: 'light',
    autoRefresh: true,
    refreshInterval: 30000
  });

  const [newIndicator, setNewIndicator] = useState({
    type: 'SMA' as const,
    period: 20,
    parameters: {}
  });

  const availableSymbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'BTC-USD', 'ETH-USD'];
  const availableTimeframes = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '30m', label: '30 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
    { value: '1w', label: '1 Week' },
    { value: '1M', label: '1 Month' }
  ];

  useEffect(() => {
    fetchChartData();
    fetchPatterns();
  }, [selectedSymbol, settings.timeframe]);

  useEffect(() => {
    if (settings.autoRefresh && isPlaying) {
      intervalRef.current = setInterval(() => {
        fetchChartData();
      }, settings.refreshInterval);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [settings.autoRefresh, settings.refreshInterval, isPlaying]);

  const fetchChartData = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `/api/v2/charting/data?symbol=${selectedSymbol}&timeframe=${settings.timeframe}&indicators=${indicators.filter(i => i.visible).map(i => i.name).join(',')}`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (!response.ok) throw new Error('Failed to fetch chart data');
      
      const data = await response.json();
      setChartData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch chart data');
    } finally {
      setLoading(false);
    }
  };

  const fetchPatterns = async () => {
    try {
      const response = await fetch(
        `/api/v2/charting/patterns?symbol=${selectedSymbol}&timeframe=${settings.timeframe}`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      if (!response.ok) throw new Error('Failed to fetch patterns');
      
      const data = await response.json();
      setPatterns(data);
    } catch (err) {
      console.error('Error fetching patterns:', err);
    }
  };

  const addIndicator = async () => {
    try {
      const response = await fetch('/api/v2/charting/indicators', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          symbol: selectedSymbol,
          type: newIndicator.type,
          period: newIndicator.period,
          parameters: newIndicator.parameters
        })
      });
      
      if (!response.ok) throw new Error('Failed to add indicator');
      
      const indicator = await response.json();
      setIndicators(prev => [...prev, {
        ...indicator,
        visible: true,
        color: getRandomColor()
      }]);
      
      // Reset form
      setNewIndicator({
        type: 'SMA',
        period: 20,
        parameters: {}
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add indicator');
    }
  };

  const removeIndicator = (index: number) => {
    setIndicators(prev => prev.filter((_, i) => i !== index));
  };

  const toggleIndicator = (index: number) => {
    setIndicators(prev => 
      prev.map((indicator, i) => 
        i === index ? { ...indicator, visible: !indicator.visible } : indicator
      )
    );
  };

  const exportChart = async () => {
    try {
      const response = await fetch('/api/v2/charting/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          symbol: selectedSymbol,
          timeframe: settings.timeframe,
          chartType: settings.chartType,
          indicators: indicators.filter(i => i.visible)
        })
      });
      
      if (!response.ok) throw new Error('Failed to export chart');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedSymbol}_${settings.timeframe}_chart.png`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export chart');
    }
  };

  const getRandomColor = () => {
    const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const getPatternColor = (pattern: Pattern) => {
    return pattern.bullish ? 'text-green-600' : 'text-red-600';
  };

  const getPatternIcon = (pattern: Pattern) => {
    return pattern.bullish ? <TrendingUpIcon className="h-4 w-4" /> : <TrendingDownIcon className="h-4 w-4" />;
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toString();
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      chartRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const currentPrice = chartData.length > 0 ? chartData[chartData.length - 1].close : 0;
  const previousPrice = chartData.length > 1 ? chartData[chartData.length - 2].close : 0;
  const priceChange = currentPrice - previousPrice;
  const priceChangePercent = previousPrice > 0 ? (priceChange / previousPrice) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <BarChart3Icon className="h-8 w-8" />
          <div>
            <h1 className="text-3xl font-bold">Advanced Charting</h1>
            <p className="text-gray-600">Professional trading charts with technical analysis</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center space-x-2"
          >
            {isPlaying ? <PauseIcon className="h-4 w-4" /> : <PlayIcon className="h-4 w-4" />}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </Button>
          <Button variant="outline" onClick={exportChart}>
            <DownloadIcon className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangleIcon className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Symbol and Price Info */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableSymbols.map(symbol => (
                    <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div>
                <p className="text-2xl font-bold">{formatPrice(currentPrice)}</p>
                <div className={`flex items-center space-x-2 ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {priceChange >= 0 ? <TrendingUpIcon className="h-4 w-4" /> : <TrendingDownIcon className="h-4 w-4" />}
                  <span>{priceChange >= 0 ? '+' : ''}{formatPrice(priceChange)}</span>
                  <span>({priceChange >= 0 ? '+' : ''}{priceChangePercent.toFixed(2)}%)</span>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Select value={settings.timeframe} onValueChange={(value: any) => setSettings(prev => ({ ...prev, timeframe: value }))}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableTimeframes.map(tf => (
                    <SelectItem key={tf.value} value={tf.value}>{tf.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button variant="outline" onClick={fetchChartData} disabled={loading}>
                <RefreshCwIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Indicators Interface */}
      <TechnicalsInterface 
        selectedAssetType={selectedAssetType}
        onIndicatorToggle={async (indicatorName: string, enabled: boolean, parameters?: any) => {
          if (enabled) {
            setEnabledIndicators(prev => [...prev, indicatorName]);
            try {
              setLoading(true);
              
              // Determine asset type from symbol
              const assetType = getAssetType(selectedSymbol);
              
              // Prepare request data
              const requestData = {
                symbol: selectedSymbol,
                timeframe: settings.timeframe,
                indicator_type: indicatorName,
                parameters: parameters || {},
                start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), // Last 30 days
                end_date: new Date().toISOString()
              };
              
              // Call appropriate backend endpoint based on asset type
              const endpoint = `/api/indicators/${assetType}`;
              const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
              });
              
              if (!response.ok) {
                throw new Error(`Failed to calculate ${indicatorName}: ${response.statusText}`);
              }
              
              const result = await response.json();
              
              // Add indicator to chart data
              setChartData(prevData => 
                prevData.map(dataPoint => ({
                  ...dataPoint,
                  indicators: {
                    ...dataPoint.indicators,
                    [indicatorName]: result.values[dataPoint.timestamp] || null
                  }
                }))
              );
              
              // Add to active indicators
              setIndicators(prev => [...prev, {
                name: indicatorName,
                type: indicatorName as any,
                period: parameters?.period || 14,
                parameters: parameters || {},
                visible: true,
                color: getIndicatorColor(indicatorName)
              }]);
              
              console.log(`Successfully added indicator: ${indicatorName}`);
            } catch (error) {
              console.error(`Error adding indicator ${indicatorName}:`, error);
              setError(`Failed to add ${indicatorName}: ${error instanceof Error ? error.message : 'Unknown error'}`);
            } finally {
              setLoading(false);
            }
          } else {
            // Remove indicator
            setEnabledIndicators(prev => prev.filter(id => id !== indicatorName));
            setIndicators(prev => prev.filter(ind => ind.name !== indicatorName));
            setChartData(prevData => 
              prevData.map(dataPoint => {
                const { [indicatorName]: removed, ...remainingIndicators } = dataPoint.indicators || {};
                return {
                  ...dataPoint,
                  indicators: remainingIndicators
                };
              })
            );
            console.log(`Removed indicator: ${indicatorName}`);
          }
        }}
        enabledIndicators={enabledIndicators}
        onAssetTypeChange={(assetType: string) => {
          setSelectedAssetType(assetType as 'crypto' | 'stock' | 'forex' | 'futures' | 'index');
        }}
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Chart */}
        <div className="lg:col-span-3">
          <Card className="h-[600px]">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <CandlestickChartIcon className="h-5 w-5" />
                  <span>{selectedSymbol} Chart</span>
                </CardTitle>
                <div className="flex items-center space-x-2">
                  <Select value={settings.chartType} onValueChange={(value: any) => setSettings(prev => ({ ...prev, chartType: value }))}>
                    <SelectTrigger className="w-40">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="CANDLESTICK">Candlestick</SelectItem>
                      <SelectItem value="LINE">Line</SelectItem>
                      <SelectItem value="BAR">Bar</SelectItem>
                      <SelectItem value="AREA">Area</SelectItem>
                      <SelectItem value="HEIKIN_ASHI">Heikin Ashi</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button variant="outline" size="sm" onClick={toggleFullscreen}>
                    {isFullscreen ? <MinimizeIcon className="h-4 w-4" /> : <MaximizeIcon className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-4 h-full">
              <div ref={chartRef} className="w-full h-full bg-gray-50 rounded-lg flex items-center justify-center">
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <RefreshCwIcon className="h-6 w-6 animate-spin" />
                    <span>Loading chart data...</span>
                  </div>
                ) : chartData.length > 0 ? (
                  <div className="w-full h-full relative">
                    {/* Simulated Chart Area */}
                    <div className="absolute inset-0 bg-gray-50 rounded-lg">
                      <div className="p-4 h-full flex flex-col">
                        <div className="flex-1 relative">
                          {/* Price Grid Lines */}
                          {settings.showGrid && (
                            <div className="absolute inset-0">
                              {[...Array(10)].map((_, i) => (
                                <div key={i} className="absolute w-full border-t border-gray-200" style={{ top: `${i * 10}%` }} />
                              ))}
                              {[...Array(10)].map((_, i) => (
                                <div key={i} className="absolute h-full border-l border-gray-200" style={{ left: `${i * 10}%` }} />
                              ))}
                            </div>
                          )}
                          
                          {/* Simulated Candlesticks */}
                          <div className="absolute inset-0 flex items-end justify-around px-4">
                            {chartData.slice(-20).map((candle, index) => {
                              const isGreen = candle.close > candle.open;
                              const height = Math.random() * 60 + 20;
                              return (
                                <div key={index} className="flex flex-col items-center">
                                  <div 
                                    className={`w-2 ${isGreen ? 'bg-green-500' : 'bg-red-500'} rounded-sm`}
                                    style={{ height: `${height}%` }}
                                  />
                                  <div className="w-px bg-gray-400 h-2" />
                                </div>
                              );
                            })}
                          </div>
                          
                          {/* Indicator Lines */}
                          {indicators.filter(i => i.visible).map((indicator, index) => (
                            <div key={index} className="absolute inset-0">
                              <svg className="w-full h-full">
                                <path
                                  d={`M 0,${50 + index * 10} Q 25,${40 + index * 10} 50,${60 + index * 10} T 100,${50 + index * 10}`}
                                  stroke={indicator.color}
                                  strokeWidth="2"
                                  fill="none"
                                  vectorEffect="non-scaling-stroke"
                                />
                              </svg>
                            </div>
                          ))}
                        </div>
                        
                        {/* Volume Chart */}
                        {settings.showVolume && (
                          <div className="h-20 border-t border-gray-200 mt-2 pt-2">
                            <div className="flex items-end justify-around h-full">
                              {chartData.slice(-20).map((candle, index) => (
                                <div 
                                  key={index}
                                  className="w-2 bg-blue-300 rounded-t-sm"
                                  style={{ height: `${Math.random() * 80 + 20}%` }}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Chart Info Overlay */}
                    <div className="absolute top-4 left-4 bg-white bg-opacity-90 rounded-lg p-3 shadow-sm">
                      <div className="text-sm space-y-1">
                        <div>O: {formatPrice(currentPrice * 0.98)}</div>
                        <div>H: {formatPrice(currentPrice * 1.02)}</div>
                        <div>L: {formatPrice(currentPrice * 0.96)}</div>
                        <div>C: {formatPrice(currentPrice)}</div>
                        <div>V: {formatVolume(Math.floor(Math.random() * 10000000))}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500">
                    <BarChart3Icon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                    <p>No chart data available</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Technical Indicators */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Technical Indicators</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Add Indicator */}
              <div className="space-y-2">
                <Label>Add Indicator</Label>
                <Select value={newIndicator.type} onValueChange={(value: any) => setNewIndicator(prev => ({ ...prev, type: value }))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SMA">Simple Moving Average</SelectItem>
                    <SelectItem value="EMA">Exponential Moving Average</SelectItem>
                    <SelectItem value="RSI">RSI</SelectItem>
                    <SelectItem value="MACD">MACD</SelectItem>
                    <SelectItem value="BOLLINGER_BANDS">Bollinger Bands</SelectItem>
                    <SelectItem value="STOCHASTIC">Stochastic</SelectItem>
                    <SelectItem value="WILLIAMS_R">Williams %R</SelectItem>
                    <SelectItem value="CCI">CCI</SelectItem>
                    <SelectItem value="ADX">ADX</SelectItem>
                  </SelectContent>
                </Select>
                <Input
                  type="number"
                  placeholder="Period"
                  value={newIndicator.period.toString()}
                  onChange={(e) => setNewIndicator(prev => ({ ...prev, period: parseInt(e.target.value) || 20 }))}
                />
                <Button onClick={addIndicator} className="w-full">
                  Add Indicator
                </Button>
              </div>
              
              {/* Active Indicators */}
              <div className="space-y-2">
                <Label>Active Indicators</Label>
                {indicators.map((indicator, index) => (
                  <div key={index} className="flex items-center justify-between p-2 border rounded">
                    <div className="flex items-center space-x-2">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: indicator.color }}
                      />
                      <span className="text-sm font-medium">{indicator.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {indicator.period}
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleIndicator(index)}
                        className={indicator.visible ? 'text-green-600' : 'text-gray-400'}
                      >
                        {indicator.visible ? '👁️' : '👁️‍🗨️'}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeIndicator(index)}
                        className="text-red-600"
                      >
                        ×
                      </Button>
                    </div>
                  </div>
                ))}
                
                {indicators.length === 0 && (
                  <p className="text-sm text-gray-500 text-center py-4">
                    No indicators added
                  </p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Pattern Recognition */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Pattern Recognition</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {patterns.map((pattern, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div className={getPatternColor(pattern)}>
                          {getPatternIcon(pattern)}
                        </div>
                        <span className="font-medium text-sm">{pattern.type}</span>
                      </div>
                      <Badge variant={pattern.confidence > 0.7 ? 'default' : 'secondary'}>
                        {Math.round(pattern.confidence * 100)}%
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 mb-2">{pattern.description}</p>
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>Start: {new Date(pattern.startTime).toLocaleDateString()}</span>
                      <span>End: {new Date(pattern.endTime).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
                
                {patterns.length === 0 && (
                  <div className="text-center py-8">
                    <InfoIcon className="h-8 w-8 mx-auto mb-2 text-gray-400" />
                    <p className="text-sm text-gray-500">No patterns detected</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Chart Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center">
                <SettingsIcon className="h-4 w-4 mr-2" />
                Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Theme</Label>
                <Select value={settings.theme} onValueChange={(value: any) => setSettings(prev => ({ ...prev, theme: value }))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">Light</SelectItem>
                    <SelectItem value="dark">Dark</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Show Volume</Label>
                <input
                  type="checkbox"
                  checked={settings.showVolume}
                  onChange={(e) => setSettings(prev => ({ ...prev, showVolume: e.target.checked }))}
                  className="rounded"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Show Grid</Label>
                <input
                  type="checkbox"
                  checked={settings.showGrid}
                  onChange={(e) => setSettings(prev => ({ ...prev, showGrid: e.target.checked }))}
                  className="rounded"
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label>Auto Refresh</Label>
                <input
                  type="checkbox"
                  checked={settings.autoRefresh}
                  onChange={(e) => setSettings(prev => ({ ...prev, autoRefresh: e.target.checked }))}
                  className="rounded"
                />
              </div>
              
              {settings.autoRefresh && (
                <div className="space-y-2">
                  <Label>Refresh Interval (seconds)</Label>
                  <Input
                    type="number"
                    min="5"
                    max="300"
                    value={(settings.refreshInterval / 1000).toString()}
                    onChange={(e) => setSettings(prev => ({ 
                      ...prev, 
                      refreshInterval: (parseInt(e.target.value) || 30) * 1000 
                    }))}
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default AdvancedCharting;