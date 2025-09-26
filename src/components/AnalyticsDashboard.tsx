"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  PieChart, 
  LineChart,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  Eye,
  AlertTriangle
} from 'lucide-react';

interface AnalyticsMetric {
  id: string;
  title: string;
  value: string;
  change: number;
  changeType: 'positive' | 'negative' | 'neutral';
  icon: React.ReactNode;
}

interface ChartData {
  id: string;
  title: string;
  type: 'line' | 'bar' | 'pie';
  data: any[];
  timeframe: string;
}

const AnalyticsDashboard: React.FC = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['all']);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Mock analytics data
  const metrics: AnalyticsMetric[] = [
    {
      id: 'total-pnl',
      title: 'Total P&L',
      value: '$24,567.89',
      change: 12.5,
      changeType: 'positive',
      icon: <TrendingUp className="h-4 w-4" />
    },
    {
      id: 'win-rate',
      title: 'Win Rate',
      value: '68.4%',
      change: 3.2,
      changeType: 'positive',
      icon: <Activity className="h-4 w-4" />
    },
    {
      id: 'avg-trade',
      title: 'Avg Trade Size',
      value: '$1,234.56',
      change: -2.1,
      changeType: 'negative',
      icon: <BarChart3 className="h-4 w-4" />
    },
    {
      id: 'sharpe-ratio',
      title: 'Sharpe Ratio',
      value: '1.87',
      change: 0.15,
      changeType: 'positive',
      icon: <PieChart className="h-4 w-4" />
    }
  ];

  const charts: ChartData[] = [
    {
      id: 'performance',
      title: 'Portfolio Performance',
      type: 'line',
      data: [],
      timeframe: '30d'
    },
    {
      id: 'asset-allocation',
      title: 'Asset Allocation',
      type: 'pie',
      data: [],
      timeframe: 'current'
    },
    {
      id: 'trade-volume',
      title: 'Trading Volume',
      type: 'bar',
      data: [],
      timeframe: '7d'
    }
  ];

  const handleRefresh = async () => {
    setIsRefreshing(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsRefreshing(false);
  };

  const handleExport = () => {
    // Simulate export functionality
    console.log('Exporting analytics data...');
  };

  const getChangeColor = (changeType: string) => {
    switch (changeType) {
      case 'positive': return 'text-green-600';
      case 'negative': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getChangeIcon = (changeType: string) => {
    switch (changeType) {
      case 'positive': return <TrendingUp className="h-3 w-3" />;
      case 'negative': return <TrendingDown className="h-3 w-3" />;
      default: return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h2>
          <p className="text-gray-600">Comprehensive trading performance analysis</p>
        </div>
        
        <div className="flex flex-wrap gap-2">
          <select 
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="1d">1 Day</option>
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
            <option value="90d">90 Days</option>
            <option value="1y">1 Year</option>
          </select>
          
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric) => (
          <Card key={metric.id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    {metric.icon}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">{metric.title}</p>
                    <p className="text-2xl font-bold text-gray-900">{metric.value}</p>
                  </div>
                </div>
              </div>
              
              <div className={`flex items-center mt-2 ${getChangeColor(metric.changeType)}`}>
                {getChangeIcon(metric.changeType)}
                <span className="text-sm font-medium ml-1">
                  {metric.change > 0 ? '+' : ''}{metric.change}%
                </span>
                <span className="text-xs text-gray-500 ml-2">vs last period</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {charts.map((chart) => (
          <Card key={chart.id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{chart.title}</CardTitle>
                <div className="flex items-center space-x-2">
                  <Badge variant="secondary">{chart.timeframe}</Badge>
                  <Button variant="ghost" size="sm">
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <LineChart className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">Chart visualization</p>
                  <p className="text-xs text-gray-400">Data will be rendered here</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Performance Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-yellow-500" />
            Performance Insights
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
              <div>
                <p className="font-medium text-gray-900">Strong Performance</p>
                <p className="text-sm text-gray-600">Your portfolio has outperformed the market by 8.3% this month.</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2"></div>
              <div>
                <p className="font-medium text-gray-900">Risk Management</p>
                <p className="text-sm text-gray-600">Consider reducing exposure to high-volatility assets to maintain current Sharpe ratio.</p>
              </div>
            </div>
            
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
              <div>
                <p className="font-medium text-gray-900">Diversification Opportunity</p>
                <p className="text-sm text-gray-600">Adding international equities could improve portfolio stability.</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Risk Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">VaR (95%)</span>
                <span className="font-medium">-$2,456</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Max Drawdown</span>
                <span className="font-medium">-8.7%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Beta</span>
                <span className="font-medium">1.23</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Volatility</span>
                <span className="font-medium">15.4%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Trade Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Total Trades</span>
                <span className="font-medium">247</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Winning Trades</span>
                <span className="font-medium">169</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Avg Hold Time</span>
                <span className="font-medium">2.3 days</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Best Trade</span>
                <span className="font-medium text-green-600">+$1,234</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Market Correlation</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">S&P 500</span>
                <span className="font-medium">0.78</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">NASDAQ</span>
                <span className="font-medium">0.82</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Gold</span>
                <span className="font-medium">-0.15</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">USD Index</span>
                <span className="font-medium">0.34</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;