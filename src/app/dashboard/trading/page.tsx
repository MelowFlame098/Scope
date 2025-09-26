"use client";

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  CurrencyDollarIcon, 
  ArrowUpIcon,
  ArrowDownIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';

const TradingPage = () => {
  const { user } = useAuth();
  const [orderType, setOrderType] = useState('market');
  const [side, setSide] = useState('buy');
  const [symbol, setSymbol] = useState('');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');

  // Mock data
  const watchlist = [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 155.80, change: 2.45, changePercent: 1.60 },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 215.30, change: -5.20, changePercent: -2.36 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 295.40, change: 8.15, changePercent: 2.84 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 2650.75, change: 15.30, changePercent: 0.58 }
  ];

  const recentOrders = [
    {
      id: '1',
      symbol: 'AAPL',
      side: 'buy',
      quantity: 10,
      price: 154.25,
      status: 'filled',
      timestamp: '2024-01-15 10:30:00'
    },
    {
      id: '2',
      symbol: 'TSLA',
      side: 'sell',
      quantity: 5,
      price: 220.50,
      status: 'pending',
      timestamp: '2024-01-15 09:15:00'
    },
    {
      id: '3',
      symbol: 'MSFT',
      side: 'buy',
      quantity: 15,
      price: 290.00,
      status: 'cancelled',
      timestamp: '2024-01-14 16:45:00'
    }
  ];

  const handleSubmitOrder = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle order submission
    console.log('Order submitted:', { symbol, side, orderType, quantity, price });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled':
        return <CheckCircleIcon className="w-4 h-4 text-green-400" />;
      case 'pending':
        return <ClockIcon className="w-4 h-4 text-yellow-400" />;
      case 'cancelled':
        return <XCircleIcon className="w-4 h-4 text-red-400" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled':
        return 'bg-green-600';
      case 'pending':
        return 'bg-yellow-600';
      case 'cancelled':
        return 'bg-red-600';
      default:
        return 'bg-gray-600';
    }
  };

  return (
    <ProtectedRoute requiredPlan="free">
      <div className="min-h-screen bg-gray-900 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Trading</h1>
            <p className="text-gray-400">Execute trades and manage orders</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Order Form */}
            <div className="lg:col-span-1">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center">
                    <CurrencyDollarIcon className="w-5 h-5 mr-2" />
                    Place Order
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleSubmitOrder} className="space-y-4">
                    {/* Symbol */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Symbol
                      </label>
                      <Input
                        type="text"
                        placeholder="e.g., AAPL"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                        className="bg-gray-700 border-gray-600 text-white"
                      />
                    </div>

                    {/* Side */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Side
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          type="button"
                          variant={side === 'buy' ? 'default' : 'outline'}
                          onClick={() => setSide('buy')}
                          className={side === 'buy' ? 'bg-green-600 hover:bg-green-700' : 'border-gray-600 text-gray-300 hover:bg-gray-700'}
                        >
                          <ArrowUpIcon className="w-4 h-4 mr-1" />
                          Buy
                        </Button>
                        <Button
                          type="button"
                          variant={side === 'sell' ? 'default' : 'outline'}
                          onClick={() => setSide('sell')}
                          className={side === 'sell' ? 'bg-red-600 hover:bg-red-700' : 'border-gray-600 text-gray-300 hover:bg-gray-700'}
                        >
                          <ArrowDownIcon className="w-4 h-4 mr-1" />
                          Sell
                        </Button>
                      </div>
                    </div>

                    {/* Order Type */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Order Type
                      </label>
                      <Select value={orderType} onValueChange={setOrderType}>
                        <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-700 border-gray-600">
                          <SelectItem value="market">Market</SelectItem>
                          <SelectItem value="limit">Limit</SelectItem>
                          <SelectItem value="stop">Stop</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Quantity */}
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Quantity
                      </label>
                      <Input
                        type="number"
                        placeholder="0"
                        value={quantity}
                        onChange={(e) => setQuantity(e.target.value)}
                        className="bg-gray-700 border-gray-600 text-white"
                      />
                    </div>

                    {/* Price (for limit orders) */}
                    {orderType === 'limit' && (
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Price
                        </label>
                        <Input
                          type="number"
                          step="0.01"
                          placeholder="0.00"
                          value={price}
                          onChange={(e) => setPrice(e.target.value)}
                          className="bg-gray-700 border-gray-600 text-white"
                        />
                      </div>
                    )}

                    <Button
                      type="submit"
                      className={`w-full ${
                        side === 'buy' 
                          ? 'bg-green-600 hover:bg-green-700' 
                          : 'bg-red-600 hover:bg-red-700'
                      }`}
                    >
                      Place {side.charAt(0).toUpperCase() + side.slice(1)} Order
                    </Button>
                  </form>
                </CardContent>
              </Card>
            </div>

            {/* Watchlist and Orders */}
            <div className="lg:col-span-2">
              <Tabs defaultValue="watchlist" className="space-y-4">
                <TabsList className="bg-gray-800 border-gray-700">
                  <TabsTrigger value="watchlist" className="data-[state=active]:bg-gray-700">
                    Watchlist
                  </TabsTrigger>
                  <TabsTrigger value="orders" className="data-[state=active]:bg-gray-700">
                    Recent Orders
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="watchlist">
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-white">Market Watch</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {watchlist.map((stock) => (
                          <div
                            key={stock.symbol}
                            className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg hover:bg-gray-700 cursor-pointer transition-colors"
                            onClick={() => setSymbol(stock.symbol)}
                          >
                            <div>
                              <div className="font-medium text-white">{stock.symbol}</div>
                              <div className="text-sm text-gray-400">{stock.name}</div>
                            </div>
                            <div className="text-right">
                              <div className="font-medium text-white">${stock.price.toFixed(2)}</div>
                              <div className={`text-sm flex items-center ${
                                stock.change >= 0 ? 'text-green-400' : 'text-red-400'
                              }`}>
                                {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.changePercent}%)
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="orders">
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-white">Recent Orders</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {recentOrders.map((order) => (
                          <div key={order.id} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                            <div className="flex items-center space-x-3">
                              <div className={`w-2 h-2 rounded-full ${getStatusColor(order.status)}`} />
                              <div>
                                <div className="font-medium text-white">
                                  {order.side.toUpperCase()} {order.quantity} {order.symbol}
                                </div>
                                <div className="text-sm text-gray-400">
                                  @ ${order.price.toFixed(2)} • {order.timestamp}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center space-x-2">
                              {getStatusIcon(order.status)}
                              <Badge className={`${getStatusColor(order.status)} text-white`}>
                                {order.status}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  );
};

export default TradingPage;