"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ChartBarIcon,
  StarIcon,
  PlusIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  BoltIcon,
  CpuChipIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import RealTimeMarketData from '../../RealTimeMarketData';

interface HubAreaProps {
  user: any;
}

export const HubArea: React.FC<HubAreaProps> = ({ user }) => {
  const [selectedAsset, setSelectedAsset] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1D');
  const [watchlist, setWatchlist] = useState([
    { symbol: 'AAPL', name: 'Apple Inc.', price: 185.92, change: 2.34, changePercent: 1.28, volume: '52.3M' },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.50, change: -5.67, changePercent: -2.23, volume: '89.1M' },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.85, change: 4.12, changePercent: 1.10, volume: '31.2M' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.56, change: 1.89, changePercent: 1.34, volume: '28.7M' },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 875.28, change: 15.67, changePercent: 1.82, volume: '45.8M' }
  ]);

  const [indicators, setIndicators] = useState([
    { name: 'RSI', value: 67.8, signal: 'neutral', enabled: true },
    { name: 'MACD', value: 2.34, signal: 'bullish', enabled: true },
    { name: 'BB', value: 'Upper', signal: 'overbought', enabled: false },
    { name: 'SMA 20', value: 182.45, signal: 'bullish', enabled: true },
    { name: 'Volume', value: '125%', signal: 'high', enabled: true }
  ]);

  const [aiInsights, setAiInsights] = useState([
    {
      type: 'prediction',
      confidence: 78,
      message: 'Strong bullish momentum expected for AAPL in next 2-4 hours',
      timeframe: '4H',
      factors: ['Volume surge', 'Technical breakout', 'Positive sentiment']
    },
    {
      type: 'risk',
      confidence: 65,
      message: 'Market volatility increasing - consider position sizing',
      timeframe: '1D',
      factors: ['VIX rising', 'Economic data pending', 'Options expiry']
    }
  ]);

  const timeframes = ['1m', '5m', '15m', '1H', '4H', '1D', '1W', '1M'];

  const toggleIndicator = (index: number) => {
    setIndicators(prev => prev.map((ind, i) => 
      i === index ? { ...ind, enabled: !ind.enabled } : ind
    ));
  };

  const addToWatchlist = () => {
    // TODO: Implement add to watchlist functionality
    setWatchlist(prev => [...prev, {
      symbol: 'NEW',
      name: 'New Symbol',
      price: 0,
      change: 0,
      changePercent: 0,
      volume: '0'
    }]);
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      case 'neutral': return 'text-yellow-400';
      case 'overbought': return 'text-orange-400';
      case 'oversold': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  const getChangeColor = (change: number) => {
    return change >= 0 ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Real-Time Market Data Section */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <RealTimeMarketData 
          symbols={['BTC-USD', 'ETH-USD', 'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'NVDA']}
          autoRefresh={true}
          refreshInterval={3000}
        />
      </div>

      {/* Watchlist Section */}
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center">
              <StarIcon className="w-5 h-5 mr-2" />
              Watchlist
            </CardTitle>
            <Button size="sm" onClick={addToWatchlist}>
              <PlusIcon className="w-4 h-4 mr-1" />
              Add
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {watchlist.map((asset) => (
              <div
                key={asset.symbol}
                className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors ${
                  selectedAsset === asset.symbol 
                    ? 'bg-blue-600/20 border border-blue-500' 
                    : 'bg-gray-700/50 hover:bg-gray-700'
                }`}
                onClick={() => setSelectedAsset(asset.symbol)}
              >
                <div>
                  <div className="font-medium text-white">{asset.symbol}</div>
                  <div className="text-sm text-gray-400 truncate">{asset.name}</div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-white">${asset.price}</div>
                  <div className={`text-sm flex items-center ${getChangeColor(asset.change)}`}>
                    {asset.change >= 0 ? (
                      <ArrowUpIcon className="w-3 h-3 mr-1" />
                    ) : (
                      <ArrowDownIcon className="w-3 h-3 mr-1" />
                    )}
                    {asset.changePercent}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Chart Section */}
      <Card className="bg-gray-800 border-gray-700 flex-1">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center">
              <ChartBarIcon className="w-5 h-5 mr-2" />
              {selectedAsset} Chart
            </CardTitle>
            <div className="flex items-center space-x-2">
              {timeframes.map((tf) => (
                <Button
                  key={tf}
                  variant={timeframe === tf ? "default" : "outline"}
                  size="sm"
                  onClick={() => setTimeframe(tf)}
                  className="h-7 px-2 text-xs"
                >
                  {tf}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-0 h-full">
          <div className="bg-gray-900 rounded-lg p-4 h-64 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <ChartBarIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Interactive Chart for {selectedAsset}</p>
              <p className="text-sm">Timeframe: {timeframe}</p>
              <p className="text-xs mt-2">Chart integration pending</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Indicators & AI Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Technical Indicators */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-white flex items-center">
              <BoltIcon className="w-5 h-5 mr-2" />
              Indicators
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="space-y-3">
              {indicators.map((indicator, index) => (
                <div key={indicator.name} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleIndicator(index)}
                      className={`h-6 w-6 p-0 ${indicator.enabled ? 'text-green-400' : 'text-gray-500'}`}
                    >
                      <EyeIcon className="w-4 h-4" />
                    </Button>
                    <span className="text-white font-medium">{indicator.name}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-white">{indicator.value}</div>
                    <div className={`text-xs ${getSignalColor(indicator.signal)}`}>
                      {indicator.signal}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* AI Insights */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-white flex items-center">
              <CpuChipIcon className="w-5 h-5 mr-2" />
              AI Insights
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="space-y-3">
              {aiInsights.map((insight, index) => (
                <div key={index} className="bg-gray-700/50 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <Badge 
                      className={`${
                        insight.type === 'prediction' ? 'bg-blue-600' : 'bg-orange-600'
                      } text-white`}
                    >
                      {insight.type}
                    </Badge>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-gray-400">{insight.timeframe}</span>
                      <Badge variant="outline" className="text-xs">
                        {insight.confidence}%
                      </Badge>
                    </div>
                  </div>
                  <p className="text-white text-sm mb-2">{insight.message}</p>
                  <div className="flex flex-wrap gap-1">
                    {insight.factors.map((factor, i) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {factor}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default HubArea;