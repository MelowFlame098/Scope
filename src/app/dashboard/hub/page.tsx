'use client';

import React, { useState } from 'react';
import Watchlist from '@/components/Watchlist';
import TradingChart from '@/components/TradingChart';
import AIInsights from '@/components/AIInsights';
import FloatingAIAssistant from '@/components/FloatingAIAssistant';
import TechnicalsInterface from '@/components/TechnicalsInterface';

export default function HubPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('AAPL');
  const [timeframe, setTimeframe] = useState<string>('1D');

  const timeframes = [
    { label: '1m', value: '1m' },
    { label: '5m', value: '5m' },
    { label: '10m', value: '10m' },
    { label: '15m', value: '15m' },
    { label: '30m', value: '30m' },
    { label: '1h', value: '1h' },
    { label: '2h', value: '2h' },
    { label: '6h', value: '6h' },
    { label: '12h', value: '12h' },
    { label: '1D', value: '1D' },
    { label: '1W', value: '1W' },
    { label: '1M', value: '1M' },
    { label: '6M', value: '6M' },
    { label: '1Y', value: '1Y' },
    { label: '5Y', value: '5Y' },
    { label: 'All', value: 'All' }
  ];

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left Sidebar - Watchlist */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">The Hub</h2>
          <p className="text-sm text-gray-400">Your Trading Command Center</p>
        </div>
        <div className="flex-1 overflow-hidden">
          <Watchlist />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Chart Header with Timeframe Selector */}
        <div className="bg-gray-800 border-b border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h3 className="text-lg font-semibold">{selectedSymbol}</h3>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-400">Timeframe:</span>
                <select
                  value={timeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                  className="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {timeframes.map((tf) => (
                    <option key={tf.value} value={tf.value}>
                      {tf.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            
            {/* Indicators Toggle */}
            <div className="flex items-center space-x-2">
              <TechnicalsInterface 
                selectedAssetType="crypto"
                onIndicatorToggle={(id, enabled) => {}}
                enabledIndicators={[]}
                onAssetTypeChange={() => {}}
              />
            </div>
          </div>
        </div>

        {/* Chart Area */}
        <div className="flex-1 bg-gray-900 p-4">
          <div className="h-full bg-gray-800 rounded-lg border border-gray-700">
            <TradingChart />
          </div>
        </div>

        {/* AI Insights Panel */}
        <div className="h-64 bg-gray-800 border-t border-gray-700 p-4">
          <div className="h-full bg-gray-900 rounded-lg border border-gray-700 p-4">
            <h4 className="text-lg font-semibold mb-3 text-white">AI Market Insights</h4>
            <AIInsights />
          </div>
        </div>
      </div>

      {/* Floating AI Assistant */}
      <FloatingAIAssistant />
    </div>
  );
}