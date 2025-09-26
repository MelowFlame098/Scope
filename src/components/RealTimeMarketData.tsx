import React, { useState, useEffect, useCallback } from 'react';
import { marketDataService, PriceData, MarketStats } from '../services/MarketDataService';
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Zap } from 'lucide-react';

// Inline UI Components
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white rounded-lg shadow-md border border-gray-200 ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children }: { children: React.ReactNode }) => (
  <div className="px-6 py-4 border-b border-gray-200">
    {children}
  </div>
);

const CardTitle = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <h3 className={`text-lg font-semibold text-gray-900 ${className}`}>
    {children}
  </h3>
);

const CardContent = ({ children }: { children: React.ReactNode }) => (
  <div className="px-6 py-4">
    {children}
  </div>
);

const Badge = ({ children, variant = 'default' }: { children: React.ReactNode; variant?: 'default' | 'success' | 'destructive' }) => {
  const variants = {
    default: 'bg-gray-100 text-gray-800',
    success: 'bg-green-100 text-green-800',
    destructive: 'bg-red-100 text-red-800'
  };
  
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  );
};

const Button = ({ 
  children, 
  onClick, 
  variant = 'default', 
  size = 'default',
  disabled = false 
}: { 
  children: React.ReactNode; 
  onClick?: () => void; 
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm' | 'lg';
  disabled?: boolean;
}) => {
  const variants = {
    default: 'bg-blue-600 hover:bg-blue-700 text-white',
    outline: 'border border-gray-300 bg-white hover:bg-gray-50 text-gray-700',
    ghost: 'hover:bg-gray-100 text-gray-700'
  };
  
  const sizes = {
    default: 'px-4 py-2 text-sm',
    sm: 'px-3 py-1.5 text-xs',
    lg: 'px-6 py-3 text-base'
  };
  
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]}`}
    >
      {children}
    </button>
  );
};

interface RealTimeMarketDataProps {
  symbols?: string[];
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const RealTimeMarketData: React.FC<RealTimeMarketDataProps> = ({
  symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'GOOGL', 'TSLA'],
  autoRefresh = true,
  refreshInterval = 5000
}) => {
  const [priceData, setPriceData] = useState<Record<string, PriceData>>({});
  const [marketStats, setMarketStats] = useState<MarketStats | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Initialize market data service and subscriptions
  useEffect(() => {
    const initializeService = async () => {
      try {
        setLoading(true);
        
        // Set up event listeners
        marketDataService.on('connected', () => {
          setIsConnected(true);
          setError(null);
        });
        
        marketDataService.on('disconnected', () => {
          setIsConnected(false);
        });
        
        marketDataService.on('error', (err) => {
          setError(err.message || 'Connection error');
        });
        
        marketDataService.on('priceUpdate', ({ symbol, data }) => {
          setPriceData(prev => ({
            ...prev,
            [symbol]: data
          }));
          setLastUpdate(new Date());
        });
        
        // Subscribe to symbols
        symbols.forEach(symbol => {
          marketDataService.subscribeToSymbol(symbol);
        });
        
        // Fetch initial data
        await fetchInitialData();
        
      } catch (err) {
        setError('Failed to initialize market data service');
        console.error('Market data initialization error:', err);
      } finally {
        setLoading(false);
      }
    };

    initializeService();

    // Cleanup on unmount
    return () => {
      symbols.forEach(symbol => {
        marketDataService.unsubscribeFromSymbol(symbol);
      });
      marketDataService.removeAllListeners();
    };
  }, [symbols]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(async () => {
      await fetchLatestPrices();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, symbols]);

  const fetchInitialData = async () => {
    try {
      // Fetch initial prices
      const prices = await marketDataService.fetchMultipleRealTimePrices(symbols);
      setPriceData(prices);
      
      // Fetch market stats
      const stats = await marketDataService.getMarketStats();
      setMarketStats(stats);
      
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Error fetching initial data:', err);
      setError('Failed to fetch initial market data');
    }
  };

  const fetchLatestPrices = async () => {
    try {
      const prices = await marketDataService.fetchMultipleRealTimePrices(symbols);
      setPriceData(prev => ({ ...prev, ...prices }));
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Error fetching latest prices:', err);
    }
  };

  const refreshData = useCallback(async () => {
    setLoading(true);
    await fetchInitialData();
    setLoading(false);
  }, [symbols]);

  const formatPrice = (price: number): string => {
    if (price >= 1000) {
      return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return price.toFixed(price < 1 ? 6 : 2);
  };

  const formatChange = (change: number, changePercent: number): { text: string; color: string; icon: React.ReactNode } => {
    const isPositive = change >= 0;
    return {
      text: `${isPositive ? '+' : ''}${change.toFixed(2)} (${isPositive ? '+' : ''}${changePercent.toFixed(2)}%)`,
      color: isPositive ? 'text-green-600' : 'text-red-600',
      icon: isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />
    };
  };

  const formatVolume = (volume: number): string => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toString();
  };

  const formatMarketCap = (marketCap: number): string => {
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(2)}T`;
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(2)}B`;
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(2)}M`;
    return `$${marketCap.toLocaleString()}`;
  };

  if (loading && Object.keys(priceData).length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading market data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="text-2xl font-bold text-gray-900">Real-Time Market Data</h2>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          {lastUpdate && (
            <span className="text-sm text-gray-500">
              Last update: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
        <Button onClick={refreshData} disabled={loading} variant="outline">
          <Activity className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="text-red-800">
              <strong>Error:</strong> {error}
            </div>
          </div>
        </div>
      )}

      {/* Market Overview Stats */}
      {marketStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent>
              <div className="flex items-center">
                <DollarSign className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Market Cap</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatMarketCap(marketStats.totalMarketCap)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <div className="flex items-center">
                <BarChart3 className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">24h Volume</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatVolume(marketStats.totalVolume24h)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <div className="flex items-center">
                <Activity className="h-8 w-8 text-purple-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Active Symbols</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {marketStats.activeSymbols}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent>
              <div className="flex items-center">
                <Zap className="h-8 w-8 text-yellow-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">BTC Dominance</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {marketStats.btcDominance.toFixed(1)}%
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Real-Time Price Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {symbols.map(symbol => {
          const data = priceData[symbol];
          if (!data) {
            return (
              <Card key={symbol}>
                <CardHeader>
                  <CardTitle>{symbol}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="animate-pulse">
                    <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                    <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </CardContent>
              </Card>
            );
          }

          const changeInfo = formatChange(data.change, data.changePercent);
          
          return (
            <Card key={symbol} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{symbol}</CardTitle>
                  <Badge variant={data.change >= 0 ? 'success' : 'destructive'}>
                    {changeInfo.icon}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <p className="text-3xl font-bold text-gray-900">
                      ${formatPrice(data.price)}
                    </p>
                    <p className={`text-sm font-medium flex items-center ${changeInfo.color}`}>
                      {changeInfo.icon}
                      <span className="ml-1">{changeInfo.text}</span>
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">24h High</p>
                      <p className="font-semibold">${formatPrice(data.high24h)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">24h Low</p>
                      <p className="font-semibold">${formatPrice(data.low24h)}</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Volume</p>
                      <p className="font-semibold">{formatVolume(data.volume)}</p>
                    </div>
                    {data.marketCap && (
                      <div>
                        <p className="text-gray-600">Market Cap</p>
                        <p className="font-semibold">{formatMarketCap(data.marketCap)}</p>
                      </div>
                    )}
                  </div>
                  
                  <div className="pt-2 border-t border-gray-200">
                    <p className="text-xs text-gray-500">
                      Updated: {new Date(data.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Top Gainers and Losers */}
      {marketStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
                Top Gainers
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {marketStats.topGainers.map((asset, index) => (
                  <div key={asset.symbol} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-sm font-medium text-gray-900 w-16">
                        {asset.symbol}
                      </span>
                      <span className="text-sm text-gray-600">
                        ${formatPrice(asset.price)}
                      </span>
                    </div>
                    <Badge variant="success">
                      +{asset.changePercent.toFixed(2)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
                Top Losers
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {marketStats.topLosers.map((asset, index) => (
                  <div key={asset.symbol} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="text-sm font-medium text-gray-900 w-16">
                        {asset.symbol}
                      </span>
                      <span className="text-sm text-gray-600">
                        ${formatPrice(asset.price)}
                      </span>
                    </div>
                    <Badge variant="destructive">
                      {asset.changePercent.toFixed(2)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};

export default RealTimeMarketData;