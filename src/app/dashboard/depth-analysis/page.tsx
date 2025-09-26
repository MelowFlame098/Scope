'use client';

import React, { useState, useEffect } from 'react';
import RealTimeMarketData from '@/components/RealTimeMarketData';

interface OrderFlowData {
  price: number;
  bidSize: number;
  askSize: number;
  volume: number;
  timestamp: string;
}

interface VolumeProfile {
  price: number;
  volume: number;
  percentage: number;
}

export default function DepthAnalysisPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('AAPL');
  const [orderFlowData, setOrderFlowData] = useState<OrderFlowData[]>([]);
  const [volumeProfile, setVolumeProfile] = useState<VolumeProfile[]>([]);
  const [marketDepth, setMarketDepth] = useState<{bids: any[], asks: any[]}>({bids: [], asks: []});

  const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'BTC', 'ETH', 'SPY', 'QQQ'];

  useEffect(() => {
    // Mock data - replace with actual WebSocket connections
    const mockOrderFlow: OrderFlowData[] = Array.from({length: 20}, (_, i) => ({
      price: 150 + (Math.random() - 0.5) * 10,
      bidSize: Math.floor(Math.random() * 1000) + 100,
      askSize: Math.floor(Math.random() * 1000) + 100,
      volume: Math.floor(Math.random() * 10000) + 1000,
      timestamp: new Date(Date.now() - i * 1000).toISOString()
    }));

    const mockVolumeProfile: VolumeProfile[] = Array.from({length: 15}, (_, i) => {
      const volume = Math.floor(Math.random() * 50000) + 10000;
      return {
        price: 145 + i * 0.5,
        volume,
        percentage: (volume / 500000) * 100
      };
    });

    const mockMarketDepth = {
      bids: Array.from({length: 10}, (_, i) => ({
        price: 149.5 - i * 0.1,
        size: Math.floor(Math.random() * 500) + 100,
        orders: Math.floor(Math.random() * 20) + 5
      })),
      asks: Array.from({length: 10}, (_, i) => ({
        price: 150.0 + i * 0.1,
        size: Math.floor(Math.random() * 500) + 100,
        orders: Math.floor(Math.random() * 20) + 5
      }))
    };

    setOrderFlowData(mockOrderFlow);
    setVolumeProfile(mockVolumeProfile);
    setMarketDepth(mockMarketDepth);
  }, [selectedSymbol]);

  return (
    <div className="h-screen bg-gray-900 text-white p-4">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Depth Analysis</h1>
            <p className="text-gray-400">Advanced Order Flow & Volume Analysis</p>
          </div>
          
          <div className="flex items-center space-x-4">
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 h-[calc(100vh-120px)]">
        {/* Order Flow */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Order Flow</h3>
          <div className="h-full overflow-y-auto">
            <div className="space-y-2">
              {orderFlowData.map((data, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-700 rounded text-sm">
                  <div className="flex items-center space-x-3">
                    <span className="font-mono text-white">${data.price.toFixed(2)}</span>
                    <div className="flex space-x-2">
                      <span className="text-green-400">B:{data.bidSize}</span>
                      <span className="text-red-400">A:{data.askSize}</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-blue-400">{data.volume.toLocaleString()}</span>
                    <span className="text-gray-400 text-xs">
                      {new Date(data.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Market Depth */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Market Depth</h3>
          <div className="h-full overflow-y-auto">
            <div className="grid grid-cols-2 gap-4">
              {/* Bids */}
              <div>
                <h4 className="text-sm font-medium text-green-400 mb-2">Bids</h4>
                <div className="space-y-1">
                  {marketDepth.bids.map((bid, index) => (
                    <div key={index} className="flex justify-between text-xs p-1 bg-green-900/20 rounded">
                      <span className="text-green-400">${bid.price.toFixed(2)}</span>
                      <span className="text-white">{bid.size}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Asks */}
              <div>
                <h4 className="text-sm font-medium text-red-400 mb-2">Asks</h4>
                <div className="space-y-1">
                  {marketDepth.asks.map((ask, index) => (
                    <div key={index} className="flex justify-between text-xs p-1 bg-red-900/20 rounded">
                      <span className="text-red-400">${ask.price.toFixed(2)}</span>
                      <span className="text-white">{ask.size}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Volume Profile */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Volume Profile</h3>
          <div className="h-full overflow-y-auto">
            <div className="space-y-2">
              {volumeProfile.map((profile, index) => (
                <div key={index} className="relative">
                  <div className="flex justify-between items-center text-sm mb-1">
                    <span className="text-white font-mono">${profile.price.toFixed(2)}</span>
                    <span className="text-blue-400">{profile.volume.toLocaleString()}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${profile.percentage}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Real-time Market Data */}
        <div className="lg:col-span-2 xl:col-span-3 bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Real-time Market Data</h3>
          <div className="h-full">
            <RealTimeMarketData symbols={[selectedSymbol]} />
          </div>
        </div>

        {/* Volume Analysis */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Volume Analysis</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded p-3">
                <div className="text-xs text-gray-400 mb-1">Total Volume</div>
                <div className="text-lg font-semibold text-white">2.4M</div>
                <div className="text-xs text-green-400">+12.5%</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-xs text-gray-400 mb-1">Avg Volume</div>
                <div className="text-lg font-semibold text-white">1.8M</div>
                <div className="text-xs text-blue-400">20D Avg</div>
              </div>
            </div>
            
            <div className="bg-gray-700 rounded p-3">
              <div className="text-xs text-gray-400 mb-2">Volume Distribution</div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Buy Volume</span>
                  <span className="text-green-400">58%</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full" style={{ width: '58%' }}></div>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Sell Volume</span>
                  <span className="text-red-400">42%</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div className="bg-red-500 h-2 rounded-full" style={{ width: '42%' }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Options Flow */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Options Flow</h3>
          <div className="space-y-3">
            <div className="bg-gray-700 rounded p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-400">Call/Put Ratio</span>
                <span className="text-lg font-semibold text-white">1.24</span>
              </div>
              <div className="text-xs text-green-400">Bullish sentiment</div>
            </div>
            
            <div className="bg-gray-700 rounded p-3">
              <div className="text-xs text-gray-400 mb-2">Large Trades (&gt;$100k)</div>
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span>155C 1/19</span>
                  <span className="text-green-400">+$250k</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>150P 1/19</span>
                  <span className="text-red-400">+$180k</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span>160C 2/16</span>
                  <span className="text-green-400">+$320k</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Market Sentiment */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <h3 className="text-lg font-semibold mb-4 text-white">Market Sentiment</h3>
          <div className="space-y-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400 mb-2">72</div>
              <div className="text-sm text-gray-400">Fear & Greed Index</div>
              <div className="text-xs text-green-400">Greed</div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">RSI (14)</span>
                <span className="text-white">68.5</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">MACD</span>
                <span className="text-green-400">+2.34</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-400">Bollinger %B</span>
                <span className="text-yellow-400">0.78</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}