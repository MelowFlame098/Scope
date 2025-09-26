"use client";

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  BanknotesIcon,
  UserGroupIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  BellIcon,
  CogIcon,
  EyeIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';

interface DashboardProps {
  user: any;
}

interface PortfolioData {
  totalValue: number;
  dailyChange: number;
  dailyChangePercent: number;
  assets: Array<{
    symbol: string;
    name: string;
    value: number;
    change: number;
    changePercent: number;
    allocation: number;
  }>;
}

interface MarketData {
  indices: Array<{
    name: string;
    value: number;
    change: number;
    changePercent: number;
  }>;
  topGainers: Array<{
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
  }>;
  topLosers: Array<{
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
  }>;
}

const Dashboard: React.FC<DashboardProps> = ({ user }) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  // Mock data for demonstration
  useEffect(() => {
    const loadDashboardData = async () => {
      setIsLoading(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setPortfolioData({
        totalValue: 125430.50,
        dailyChange: 2340.25,
        dailyChangePercent: 1.89,
        assets: [
          {
            symbol: 'AAPL',
            name: 'Apple Inc.',
            value: 45230.50,
            change: 1250.30,
            changePercent: 2.84,
            allocation: 36.1
          },
          {
            symbol: 'GOOGL',
            name: 'Alphabet Inc.',
            value: 32150.75,
            change: 890.45,
            changePercent: 2.85,
            allocation: 25.6
          },
          {
            symbol: 'MSFT',
            name: 'Microsoft Corp.',
            value: 28940.25,
            change: 340.50,
            changePercent: 1.19,
            allocation: 23.1
          },
          {
            symbol: 'TSLA',
            name: 'Tesla Inc.',
            value: 19109.00,
            change: -141.00,
            changePercent: -0.73,
            allocation: 15.2
          }
        ]
      });
      
      setMarketData({
        indices: [
          { name: 'S&P 500', value: 4567.89, change: 23.45, changePercent: 0.52 },
          { name: 'NASDAQ', value: 14234.56, change: 89.12, changePercent: 0.63 },
          { name: 'DOW', value: 34567.12, change: -45.67, changePercent: -0.13 }
        ],
        topGainers: [
          { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 456.78, change: 23.45, changePercent: 5.42 },
          { symbol: 'AMD', name: 'Advanced Micro Devices', price: 123.45, change: 5.67, changePercent: 4.81 },
          { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 3456.78, change: 123.45, changePercent: 3.70 }
        ],
        topLosers: [
          { symbol: 'META', name: 'Meta Platforms Inc.', price: 234.56, change: -12.34, changePercent: -5.00 },
          { symbol: 'NFLX', name: 'Netflix Inc.', price: 345.67, change: -15.23, changePercent: -4.22 },
          { symbol: 'PYPL', name: 'PayPal Holdings Inc.', price: 67.89, change: -2.34, changePercent: -3.33 }
        ]
      });
      
      setIsLoading(false);
    };
    
    loadDashboardData();
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="bg-gray-800 border-gray-700">
              <CardContent className="p-6">
                <div className="animate-pulse">
                  <div className="h-4 bg-gray-600 rounded w-3/4 mb-2"></div>
                  <div className="h-8 bg-gray-600 rounded w-1/2"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Total Portfolio Value</p>
                <p className="text-2xl font-bold text-white">
                  {portfolioData ? formatCurrency(portfolioData.totalValue) : '$0.00'}
                </p>
              </div>
              <BanknotesIcon className="h-8 w-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Daily Change</p>
                <p className={`text-2xl font-bold ${
                  portfolioData && portfolioData.dailyChange >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolioData ? formatCurrency(portfolioData.dailyChange) : '$0.00'}
                </p>
              </div>
              {portfolioData && portfolioData.dailyChange >= 0 ? (
                <ArrowTrendingUpIcon className="h-8 w-8 text-green-400" />
              ) : (
                <ArrowTrendingDownIcon className="h-8 w-8 text-red-400" />
              )}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Daily Change %</p>
                <p className={`text-2xl font-bold ${
                  portfolioData && portfolioData.dailyChangePercent >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolioData ? formatPercent(portfolioData.dailyChangePercent) : '0.00%'}
                </p>
              </div>
              <ChartBarIcon className="h-8 w-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gray-800 border-gray-700">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-400">Assets</p>
                <p className="text-2xl font-bold text-white">
                  {portfolioData ? portfolioData.assets.length : 0}
                </p>
              </div>
              <UserGroupIcon className="h-8 w-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 bg-gray-800">
          <TabsTrigger value="overview" className="data-[state=active]:bg-blue-600">Overview</TabsTrigger>
          <TabsTrigger value="portfolio" className="data-[state=active]:bg-blue-600">Portfolio</TabsTrigger>
          <TabsTrigger value="market" className="data-[state=active]:bg-blue-600">Market</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Portfolio Allocation Chart */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Portfolio Allocation</CardTitle>
              </CardHeader>
              <CardContent>
                {portfolioData && (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={portfolioData.assets}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ symbol, allocation }) => `${symbol} ${allocation}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="allocation"
                      >
                        {portfolioData.assets.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={[
                            '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'
                          ][index % 5]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
            
            {/* Recent Activity */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Recent Activity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm text-white">Bought 10 shares of AAPL</p>
                      <p className="text-xs text-gray-400">2 hours ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm text-white">Sold 5 shares of TSLA</p>
                      <p className="text-xs text-gray-400">1 day ago</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <div className="flex-1">
                      <p className="text-sm text-white">Dividend received from MSFT</p>
                      <p className="text-xs text-gray-400">3 days ago</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="portfolio" className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Portfolio Holdings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {portfolioData?.assets.map((asset) => (
                  <div key={asset.symbol} className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                        <span className="text-white font-bold text-sm">{asset.symbol.slice(0, 2)}</span>
                      </div>
                      <div>
                        <p className="text-white font-medium">{asset.symbol}</p>
                        <p className="text-gray-400 text-sm">{asset.name}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-white font-medium">{formatCurrency(asset.value)}</p>
                      <p className={`text-sm ${
                        asset.change >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatCurrency(asset.change)} ({formatPercent(asset.changePercent)})
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="market" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Market Indices */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Market Indices</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {marketData?.indices.map((index) => (
                    <div key={index.name} className="flex items-center justify-between">
                      <span className="text-gray-300">{index.name}</span>
                      <div className="text-right">
                        <p className="text-white">{index.value.toLocaleString()}</p>
                        <p className={`text-sm ${
                          index.change >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {formatPercent(index.changePercent)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Top Gainers */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Top Gainers</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {marketData?.topGainers.map((stock) => (
                    <div key={stock.symbol} className="flex items-center justify-between">
                      <div>
                        <p className="text-white font-medium">{stock.symbol}</p>
                        <p className="text-gray-400 text-sm">{formatCurrency(stock.price)}</p>
                      </div>
                      <p className="text-green-400 text-sm font-medium">
                        {formatPercent(stock.changePercent)}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Top Losers */}
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Top Losers</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {marketData?.topLosers.map((stock) => (
                    <div key={stock.symbol} className="flex items-center justify-between">
                      <div>
                        <p className="text-white font-medium">{stock.symbol}</p>
                        <p className="text-gray-400 text-sm">{formatCurrency(stock.price)}</p>
                      </div>
                      <p className="text-red-400 text-sm font-medium">
                        {formatPercent(stock.changePercent)}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export { Dashboard };