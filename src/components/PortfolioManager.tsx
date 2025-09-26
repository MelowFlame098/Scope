import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  PieChart, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target,
  BarChart3,
  Settings,
  Plus,
  Minus,
  RefreshCw
} from 'lucide-react';

interface PortfolioHolding {
  id: string;
  symbol: string;
  name: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  value: number;
  pnl: number;
  pnlPercent: number;
  allocation: number;
}

interface PortfolioMetrics {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  dayChange: number;
  dayChangePercent: number;
  cash: number;
}

const PortfolioManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  // Mock portfolio data
  const portfolioMetrics: PortfolioMetrics = {
    totalValue: 125750.50,
    totalPnL: 8750.50,
    totalPnLPercent: 7.48,
    dayChange: 1250.75,
    dayChangePercent: 1.01,
    cash: 15250.00
  };

  const holdings: PortfolioHolding[] = [
    {
      id: '1',
      symbol: 'AAPL',
      name: 'Apple Inc.',
      quantity: 50,
      avgPrice: 175.25,
      currentPrice: 182.50,
      value: 9125.00,
      pnl: 362.50,
      pnlPercent: 4.14,
      allocation: 7.25
    },
    {
      id: '2',
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      quantity: 30,
      avgPrice: 340.00,
      currentPrice: 365.75,
      value: 10972.50,
      pnl: 772.50,
      pnlPercent: 7.58,
      allocation: 8.72
    },
    {
      id: '3',
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      quantity: 25,
      avgPrice: 125.50,
      currentPrice: 138.25,
      value: 3456.25,
      pnl: 318.75,
      pnlPercent: 10.16,
      allocation: 2.75
    },
    {
      id: '4',
      symbol: 'TSLA',
      name: 'Tesla Inc.',
      quantity: 15,
      avgPrice: 220.00,
      currentPrice: 195.50,
      value: 2932.50,
      pnl: -367.50,
      pnlPercent: -11.14,
      allocation: 2.33
    },
    {
      id: '5',
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      quantity: 20,
      avgPrice: 450.00,
      currentPrice: 485.75,
      value: 9715.00,
      pnl: 715.00,
      pnlPercent: 7.95,
      allocation: 7.72
    }
  ];

  const topGainers = holdings
    .filter(h => h.pnlPercent > 0)
    .sort((a, b) => b.pnlPercent - a.pnlPercent)
    .slice(0, 3);

  const topLosers = holdings
    .filter(h => h.pnlPercent < 0)
    .sort((a, b) => a.pnlPercent - b.pnlPercent)
    .slice(0, 3);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <PieChart className="h-6 w-6 text-blue-600" />
          <h2 className="text-2xl font-bold">Portfolio Manager</h2>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Portfolio Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Value</p>
                <p className="text-2xl font-bold">${portfolioMetrics.totalValue.toLocaleString()}</p>
              </div>
              <DollarSign className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total P&L</p>
                <p className={`text-2xl font-bold ${portfolioMetrics.totalPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  ${portfolioMetrics.totalPnL.toLocaleString()}
                </p>
                <p className={`text-sm ${portfolioMetrics.totalPnLPercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {portfolioMetrics.totalPnLPercent >= 0 ? '+' : ''}{portfolioMetrics.totalPnLPercent.toFixed(2)}%
                </p>
              </div>
              {portfolioMetrics.totalPnL >= 0 ? 
                <TrendingUp className="h-8 w-8 text-green-500" /> : 
                <TrendingDown className="h-8 w-8 text-red-500" />
              }
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Day Change</p>
                <p className={`text-2xl font-bold ${portfolioMetrics.dayChange >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  ${portfolioMetrics.dayChange.toLocaleString()}
                </p>
                <p className={`text-sm ${portfolioMetrics.dayChangePercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {portfolioMetrics.dayChangePercent >= 0 ? '+' : ''}{portfolioMetrics.dayChangePercent.toFixed(2)}%
                </p>
              </div>
              <BarChart3 className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Available Cash</p>
                <p className="text-2xl font-bold">${portfolioMetrics.cash.toLocaleString()}</p>
              </div>
              <Target className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="holdings">Holdings</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Top Gainers */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5 text-green-500" />
                  <span>Top Gainers</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {topGainers.map((holding) => (
                    <div key={holding.id} className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{holding.symbol}</p>
                        <p className="text-sm text-muted-foreground">{holding.name}</p>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-green-500">+{holding.pnlPercent.toFixed(2)}%</p>
                        <p className="text-sm text-muted-foreground">${holding.pnl.toFixed(2)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Top Losers */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingDown className="h-5 w-5 text-red-500" />
                  <span>Top Losers</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {topLosers.map((holding) => (
                    <div key={holding.id} className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{holding.symbol}</p>
                        <p className="text-sm text-muted-foreground">{holding.name}</p>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-red-500">{holding.pnlPercent.toFixed(2)}%</p>
                        <p className="text-sm text-muted-foreground">${holding.pnl.toFixed(2)}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Allocation Chart Placeholder */}
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Allocation</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {holdings.slice(0, 5).map((holding) => (
                  <div key={holding.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{holding.symbol}</span>
                      <span className="text-sm text-muted-foreground">{holding.allocation.toFixed(2)}%</span>
                    </div>
                    <Progress value={holding.allocation} className="h-2" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="holdings" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Current Holdings</CardTitle>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Position
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {holdings.map((holding) => (
                  <div key={holding.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <div>
                          <h4 className="font-medium">{holding.symbol}</h4>
                          <p className="text-sm text-muted-foreground">{holding.name}</p>
                        </div>
                      </div>
                      <div className="grid grid-cols-4 gap-4 mt-2 text-sm">
                        <div>
                          <p className="text-muted-foreground">Quantity</p>
                          <p className="font-medium">{holding.quantity}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Avg Price</p>
                          <p className="font-medium">${holding.avgPrice.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Current Price</p>
                          <p className="font-medium">${holding.currentPrice.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Market Value</p>
                          <p className="font-medium">${holding.value.toLocaleString()}</p>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className={`font-medium ${holding.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {holding.pnl >= 0 ? '+' : ''}${holding.pnl.toFixed(2)}
                        </p>
                        <p className={`text-sm ${holding.pnlPercent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          {holding.pnlPercent >= 0 ? '+' : ''}{holding.pnlPercent.toFixed(2)}%
                        </p>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Button variant="outline" size="sm">
                          <Plus className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="sm">
                          <Minus className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Total Return</span>
                    <span className="font-medium text-green-500">+{portfolioMetrics.totalPnLPercent.toFixed(2)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">1 Day Return</span>
                    <span className="font-medium text-green-500">+{portfolioMetrics.dayChangePercent.toFixed(2)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Best Performer</span>
                    <span className="font-medium">GOOGL (+10.16%)</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Worst Performer</span>
                    <span className="font-medium text-red-500">TSLA (-11.14%)</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Portfolio Beta</span>
                    <span className="font-medium">1.15</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Sharpe Ratio</span>
                    <span className="font-medium">1.42</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Max Drawdown</span>
                    <span className="font-medium text-red-500">-8.5%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Volatility</span>
                    <span className="font-medium">18.2%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PortfolioManager;