"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  ChartPieIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  ScaleIcon,
  BoltIcon,
  EyeIcon,
  AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline';

interface DepthAnalysisProps {
  user: any;
}

export const DepthAnalysis: React.FC<DepthAnalysisProps> = ({ user }) => {
  const [selectedAsset, setSelectedAsset] = useState('AAPL');
  const [activeTab, setActiveTab] = useState('orderbook');
  
  const [orderBookData, setOrderBookData] = useState({
    bids: [
      { price: 185.90, size: 1250, orders: 15 },
      { price: 185.89, size: 2100, orders: 23 },
      { price: 185.88, size: 1800, orders: 19 },
      { price: 185.87, size: 3200, orders: 31 },
      { price: 185.86, size: 2750, orders: 28 },
      { price: 185.85, size: 4100, orders: 42 },
      { price: 185.84, size: 1950, orders: 21 },
      { price: 185.83, size: 2850, orders: 29 }
    ],
    asks: [
      { price: 185.91, size: 1100, orders: 12 },
      { price: 185.92, size: 1950, orders: 18 },
      { price: 185.93, size: 2300, orders: 25 },
      { price: 185.94, size: 1750, orders: 17 },
      { price: 185.95, size: 2900, orders: 33 },
      { price: 185.96, size: 3400, orders: 38 },
      { price: 185.97, size: 2100, orders: 24 },
      { price: 185.98, size: 2650, orders: 27 }
    ]
  });

  const [volumeProfile, setVolumeProfile] = useState([
    { price: 186.00, volume: 45000, percentage: 12.5 },
    { price: 185.95, volume: 52000, percentage: 14.4 },
    { price: 185.90, volume: 68000, percentage: 18.9 },
    { price: 185.85, volume: 41000, percentage: 11.4 },
    { price: 185.80, volume: 38000, percentage: 10.6 },
    { price: 185.75, volume: 29000, percentage: 8.1 },
    { price: 185.70, volume: 35000, percentage: 9.7 },
    { price: 185.65, volume: 25000, percentage: 6.9 },
    { price: 185.60, volume: 28000, percentage: 7.8 }
  ]);

  const [flowMetrics, setFlowMetrics] = useState({
    buyVolume: 2450000,
    sellVolume: 1890000,
    netFlow: 560000,
    largeOrders: 145,
    avgOrderSize: 892,
    marketImpact: 0.023,
    spread: 0.01,
    depth: 0.15
  });

  const [levelTwoData, setLevelTwoData] = useState([
    { exchange: 'NASDAQ', bid: 185.89, ask: 185.91, bidSize: 500, askSize: 300 },
    { exchange: 'NYSE', bid: 185.88, ask: 185.92, bidSize: 750, askSize: 450 },
    { exchange: 'BATS', bid: 185.89, ask: 185.91, bidSize: 300, askSize: 200 },
    { exchange: 'EDGX', bid: 185.88, ask: 185.92, bidSize: 400, askSize: 350 }
  ]);

  const maxBidSize = Math.max(...orderBookData.bids.map(b => b.size));
  const maxAskSize = Math.max(...orderBookData.asks.map(a => a.size));
  const maxSize = Math.max(maxBidSize, maxAskSize);

  const getVolumeColor = (volume: number, maxVol: number) => {
    const intensity = (volume / maxVol) * 100;
    if (intensity > 80) return 'bg-red-500';
    if (intensity > 60) return 'bg-orange-500';
    if (intensity > 40) return 'bg-yellow-500';
    if (intensity > 20) return 'bg-blue-500';
    return 'bg-gray-500';
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  return (
    <div className="h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-800">
          <TabsTrigger value="orderbook">Order Book</TabsTrigger>
          <TabsTrigger value="volume">Volume Profile</TabsTrigger>
          <TabsTrigger value="flow">Order Flow</TabsTrigger>
          <TabsTrigger value="level2">Level II</TabsTrigger>
        </TabsList>

        {/* Order Book Tab */}
        <TabsContent value="orderbook" className="flex-1 mt-4">
          <Card className="bg-gray-800 border-gray-700 h-full">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-white flex items-center">
                  <ScaleIcon className="w-5 h-5 mr-2" />
                  Order Book - {selectedAsset}
                </CardTitle>
                <div className="flex items-center space-x-2">
                  <Badge className="bg-green-600">
                    Spread: $0.01
                  </Badge>
                  <Button variant="outline" size="sm">
                    <AdjustmentsHorizontalIcon className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0 h-full">
              <div className="grid grid-cols-2 gap-4 h-full">
                {/* Bids */}
                <div>
                  <h4 className="text-green-400 font-medium mb-3 flex items-center">
                    <ArrowUpIcon className="w-4 h-4 mr-1" />
                    Bids
                  </h4>
                  <div className="space-y-1">
                    {orderBookData.bids.map((bid, index) => (
                      <div key={index} className="relative">
                        <div 
                          className="absolute inset-0 bg-green-500/20 rounded"
                          style={{ width: `${(bid.size / maxSize) * 100}%` }}
                        />
                        <div className="relative flex justify-between items-center p-2 text-sm">
                          <span className="text-green-400 font-mono">${bid.price}</span>
                          <span className="text-white">{formatNumber(bid.size)}</span>
                          <span className="text-gray-400">{bid.orders}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Asks */}
                <div>
                  <h4 className="text-red-400 font-medium mb-3 flex items-center">
                    <ArrowDownIcon className="w-4 h-4 mr-1" />
                    Asks
                  </h4>
                  <div className="space-y-1">
                    {orderBookData.asks.map((ask, index) => (
                      <div key={index} className="relative">
                        <div 
                          className="absolute inset-0 bg-red-500/20 rounded"
                          style={{ width: `${(ask.size / maxSize) * 100}%` }}
                        />
                        <div className="relative flex justify-between items-center p-2 text-sm">
                          <span className="text-red-400 font-mono">${ask.price}</span>
                          <span className="text-white">{formatNumber(ask.size)}</span>
                          <span className="text-gray-400">{ask.orders}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Volume Profile Tab */}
        <TabsContent value="volume" className="flex-1 mt-4">
          <Card className="bg-gray-800 border-gray-700 h-full">
            <CardHeader className="pb-3">
              <CardTitle className="text-white flex items-center">
                <ChartPieIcon className="w-5 h-5 mr-2" />
                Volume Profile - {selectedAsset}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2">
                {volumeProfile.map((level, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <span className="text-white font-mono w-16 text-sm">
                      ${level.price}
                    </span>
                    <div className="flex-1">
                      <div className="relative h-6 bg-gray-700 rounded">
                        <div 
                          className={`absolute inset-y-0 left-0 rounded ${getVolumeColor(level.volume, Math.max(...volumeProfile.map(v => v.volume)))}`}
                          style={{ width: `${level.percentage * 5}%` }}
                        />
                        <div className="absolute inset-0 flex items-center justify-center text-xs text-white font-medium">
                          {formatNumber(level.volume)}
                        </div>
                      </div>
                    </div>
                    <span className="text-gray-400 text-sm w-12">
                      {level.percentage}%
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Order Flow Tab */}
        <TabsContent value="flow" className="flex-1 mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center">
                  <BoltIcon className="w-5 h-5 mr-2" />
                  Flow Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Buy Volume</span>
                    <span className="text-green-400 font-medium">
                      {formatNumber(flowMetrics.buyVolume)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Sell Volume</span>
                    <span className="text-red-400 font-medium">
                      {formatNumber(flowMetrics.sellVolume)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Net Flow</span>
                    <span className="text-green-400 font-medium">
                      +{formatNumber(flowMetrics.netFlow)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Large Orders</span>
                    <span className="text-white font-medium">
                      {flowMetrics.largeOrders}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Avg Order Size</span>
                    <span className="text-white font-medium">
                      {flowMetrics.avgOrderSize}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Market Impact</span>
                    <span className="text-yellow-400 font-medium">
                      {flowMetrics.marketImpact}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-white">Flow Visualization</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-300">Buy Pressure</span>
                      <span className="text-green-400">65%</span>
                    </div>
                    <Progress value={65} className="h-3" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-300">Sell Pressure</span>
                      <span className="text-red-400">35%</span>
                    </div>
                    <Progress value={35} className="h-3" />
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4 mt-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400 mb-1">
                        BULLISH
                      </div>
                      <div className="text-sm text-gray-300">
                        Strong buying momentum detected
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Level II Tab */}
        <TabsContent value="level2" className="flex-1 mt-4">
          <Card className="bg-gray-800 border-gray-700 h-full">
            <CardHeader className="pb-3">
              <CardTitle className="text-white flex items-center">
                <EyeIcon className="w-5 h-5 mr-2" />
                Level II Data - {selectedAsset}
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-2 text-gray-300">Exchange</th>
                      <th className="text-center py-2 text-gray-300">Bid</th>
                      <th className="text-center py-2 text-gray-300">Bid Size</th>
                      <th className="text-center py-2 text-gray-300">Ask Size</th>
                      <th className="text-center py-2 text-gray-300">Ask</th>
                    </tr>
                  </thead>
                  <tbody>
                    {levelTwoData.map((data, index) => (
                      <tr key={index} className="border-b border-gray-700/50">
                        <td className="py-2 text-white font-medium">{data.exchange}</td>
                        <td className="py-2 text-center text-green-400 font-mono">
                          ${data.bid}
                        </td>
                        <td className="py-2 text-center text-white">
                          {data.bidSize}
                        </td>
                        <td className="py-2 text-center text-white">
                          {data.askSize}
                        </td>
                        <td className="py-2 text-center text-red-400 font-mono">
                          ${data.ask}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DepthAnalysis;