'use client';

import React, { useState, useEffect } from 'react';
import { 
  ArrowTrendingUpIcon as TrendingUpIcon, 
  ArrowTrendingDownIcon as TrendingDownIcon, 
  CurrencyDollarIcon as DollarSignIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon as AlertTriangleIcon,
  ArrowPathIcon as RefreshCwIcon,
  PlayIcon,
  PauseIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { tradeExecutionService, TradeOrder } from '@/services/TradeExecutionService';
import { useAuth } from '@/contexts/authcontext';

// Inline UI components
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white rounded-lg border shadow-sm ${className}`}>{children}</div>
)
const CardHeader = ({ children }: { children: React.ReactNode }) => (
  <div className="p-6 pb-4">{children}</div>
)
const CardTitle = ({ children }: { children: React.ReactNode }) => (
  <h3 className="text-lg font-semibold">{children}</h3>
)
const CardContent = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`p-6 pt-0 ${className}`}>{children}</div>
)
const Button = ({ children, onClick, className = '', variant = 'default', size = 'default', disabled = false }: { children: React.ReactNode; onClick?: () => void; className?: string; variant?: string; size?: string; disabled?: boolean }) => (
  <button onClick={onClick} disabled={disabled} className={`px-4 py-2 rounded-md font-medium ${variant === 'outline' ? 'border border-gray-300 bg-white text-gray-700 hover:bg-gray-50' : variant === 'destructive' ? 'bg-red-600 text-white hover:bg-red-700' : 'bg-blue-600 text-white hover:bg-blue-700'} ${size === 'sm' ? 'px-2 py-1 text-sm' : ''} ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}>{children}</button>
)
const Input = ({ placeholder, value, onChange, className = '', type = 'text', id, step }: { placeholder?: string; value?: string | number; onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void; className?: string; type?: string; id?: string; step?: string }) => (
  <input id={id} type={type} step={step} placeholder={placeholder} value={value} onChange={onChange} className={`w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 ${className}`} />
)
const Label = ({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) => (
  <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-700 mb-1">{children}</label>
)
const Badge = ({ children, variant = 'default', className = '' }: { children: React.ReactNode; variant?: string; className?: string }) => (
  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
    variant === 'destructive' ? 'bg-red-100 text-red-800' : 
    variant === 'secondary' ? 'bg-gray-100 text-gray-800' : 
    variant === 'outline' ? 'bg-white border border-gray-300 text-gray-700' :
    'bg-blue-100 text-blue-800'
  } ${className}`}>{children}</span>
)
const Select = ({ children, value, onValueChange }: { children: React.ReactNode; value?: string; onValueChange?: (value: string) => void }) => {
  const [isOpen, setIsOpen] = React.useState(false);
  return (
    <div className="relative">
      {React.Children.map(children, child => {
        if (React.isValidElement(child)) {
          if (child.type === SelectTrigger) {
            return React.cloneElement(child, { onClick: () => setIsOpen(!isOpen) });
          }
          if (child.type === SelectContent && isOpen) {
            return React.cloneElement(child, { onSelect: (val: string) => { onValueChange?.(val); setIsOpen(false); } });
          }
        }
        return child;
      })}
    </div>
  );
}
const SelectTrigger = ({ children, onClick }: { children: React.ReactNode; onClick?: () => void }) => (
  <button onClick={onClick} className="w-full px-3 py-2 text-left border border-gray-300 rounded-md bg-white hover:bg-gray-50 flex justify-between items-center">
    {children}
    <span>▼</span>
  </button>
)
const SelectValue = ({ placeholder }: { placeholder?: string }) => (
  <span className="text-gray-500">{placeholder}</span>
)
const SelectContent = ({ children, onSelect }: { children: React.ReactNode; onSelect?: (value: string) => void }) => (
  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
    {React.Children.map(children, child => {
      if (React.isValidElement(child)) {
        return React.cloneElement(child, { onClick: () => onSelect?.(child.props.value) });
      }
      return child;
    })}
  </div>
)
const SelectItem = ({ children, value, onClick }: { children: React.ReactNode; value: string; onClick?: () => void }) => (
  <div onClick={onClick} className="px-3 py-2 hover:bg-gray-100 cursor-pointer">{children}</div>
)
const Tabs = ({ children, defaultValue }: { children: React.ReactNode; defaultValue?: string }) => (
  <div>{children}</div>
)
const TabsList = ({ children }: { children: React.ReactNode }) => (
  <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">{children}</div>
)
const TabsTrigger = ({ children, value }: { children: React.ReactNode; value: string }) => (
  <button className="px-3 py-1.5 text-sm font-medium rounded-md hover:bg-white hover:shadow-sm">{children}</button>
)
const TabsContent = ({ children, value }: { children: React.ReactNode; value: string }) => (
  <div className="mt-4">{children}</div>
)
const Alert = ({ children, variant = 'default' }: { children: React.ReactNode; variant?: string }) => (
  <div className={`p-4 border rounded-md flex items-start space-x-2 ${variant === 'destructive' ? 'border-red-200 bg-red-50' : 'border-blue-200 bg-blue-50'}`}>{children}</div>
)
const AlertDescription = ({ children }: { children: React.ReactNode }) => (
  <p className="text-sm text-red-700">{children}</p>
)

interface Order {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: 'PENDING' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED';
  timeInForce: 'DAY' | 'GTC' | 'IOC' | 'FOK';
  filledQuantity: number;
  averagePrice?: number;
  timestamp: string;
  portfolioId: string;
}

interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  dayChange: number;
  dayChangePercent: number;
}

interface TradingStrategy {
  id: string;
  name: string;
  description: string;
  isActive: boolean;
  parameters: Record<string, any>;
  performance: {
    totalReturn: number;
    winRate: number;
    sharpeRatio: number;
    maxDrawdown: number;
  };
}

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid: number;
  ask: number;
  high: number;
  low: number;
  open: number;
}

const TradingInterface: React.FC = () => {
  const { user } = useAuth();
  const [orders, setOrders] = useState<Order[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [strategies, setStrategies] = useState<TradingStrategy[]>([]);
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [queueStats, setQueueStats] = useState({ pending: 0, processing: 0, completed: 0 });

  // Order form state
  const [orderForm, setOrderForm] = useState<{
    symbol: string;
    side: 'BUY' | 'SELL';
    type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
    quantity: number;
    price: number;
    stopPrice: number;
    timeInForce: 'DAY' | 'GTC' | 'IOC' | 'FOK';
    portfolioId: string;
  }>({
    symbol: 'AAPL',
    side: 'BUY',
    type: 'MARKET',
    quantity: 0,
    price: 0,
    stopPrice: 0,
    timeInForce: 'DAY',
    portfolioId: ''
  });

  useEffect(() => {
    fetchOrders();
    fetchPositions();
    fetchStrategies();
    fetchMarketData();
    fetchQueueStats();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (autoRefresh) {
      interval = setInterval(() => {
        fetchOrders();
        fetchPositions();
        fetchMarketData();
        fetchQueueStats();
      }, 5000); // Refresh every 5 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const fetchOrders = async () => {
    try {
      if (!user?.id) return;
      const userOrders = await tradeExecutionService.getUserOrders(user.id);
      setOrders(userOrders.map(order => ({
        id: order.id,
        symbol: order.symbol,
        side: order.side.toUpperCase() as 'BUY' | 'SELL',
        type: order.type.toUpperCase() as 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT',
        quantity: order.quantity,
        price: order.price,
        stopPrice: order.stopPrice,
        status: (() => {
          const statusMap: Record<string, 'PENDING' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED'> = {
            'pending': 'PENDING',
            'filled': 'FILLED',
            'partial': 'PARTIALLY_FILLED',
            'cancelled': 'CANCELLED',
            'rejected': 'REJECTED'
          };
          return statusMap[order.status.toLowerCase()] || 'PENDING';
        })(),
        timeInForce: order.timeInForce,
        filledQuantity: (order as any).filledQuantity || 0,
        averagePrice: (order as any).averagePrice,
        timestamp: (order as any).timestamp?.toISOString() || new Date().toISOString(),
        portfolioId: (order as any).portfolioId || ''
      })));
    } catch (err) {
      console.error('Error fetching orders:', err);
    }
  };

  const fetchPositions = async () => {
    try {
      if (!user?.id) return;
      
      const response = await fetch('/api/v2/paper-trading/positions', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch positions');
      
      const data = await response.json();
      setPositions(data.positions || []);
    } catch (err) {
      console.error('Error fetching positions:', err);
      setError('Failed to fetch positions');
    }
  };

  const fetchStrategies = async () => {
    try {
      if (!user?.id) return;
      
      const response = await fetch('/api/v2/trading/strategies', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch strategies');
      
      const data = await response.json();
      setStrategies(data.strategies || []);
    } catch (err) {
      console.error('Error fetching strategies:', err);
    }
  };

  const fetchMarketData = async () => {
    try {
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META'];
      const response = await fetch('/api/v2/market-data/quotes', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbols })
      });
      
      if (!response.ok) throw new Error('Failed to fetch market data');
      
      const data = await response.json();
      const marketDataMap: Record<string, MarketData> = {};
      
      data.quotes?.forEach((quote: any) => {
        marketDataMap[quote.symbol] = {
          symbol: quote.symbol,
          price: quote.price || 0,
          change: quote.change || 0,
          changePercent: quote.changePercent || 0,
          volume: quote.volume || 0,
          bid: quote.bid || 0,
          ask: quote.ask || 0,
          high: quote.high || 0,
          low: quote.low || 0,
          open: quote.open || 0
        };
      });
      
      setMarketData(marketDataMap);
    } catch (err) {
      console.error('Error fetching market data:', err);
    }
  };

  const fetchQueueStats = async () => {
    try {
      const response = await fetch('/api/v2/paper-trading/queue-stats', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch queue stats');
      
      const data = await response.json();
      setQueueStats(data);
    } catch (err) {
      console.error('Error fetching queue stats:', err);
    }
  };

  const submitOrder = async () => {
    try {
      setLoading(true);
      setError(null);
      
      if (!user?.id) {
        throw new Error('User not authenticated');
      }

      // Validate order form
      if (!orderForm.symbol || orderForm.quantity <= 0) {
        throw new Error('Please enter valid symbol and quantity');
      }

      if (orderForm.type === 'LIMIT' && orderForm.price <= 0) {
        throw new Error('Please enter valid limit price');
      }

      if ((orderForm.type === 'STOP' || orderForm.type === 'STOP_LIMIT') && orderForm.stopPrice <= 0) {
        throw new Error('Please enter valid stop price');
      }

      const orderData = {
        symbol: orderForm.symbol.toUpperCase(),
        side: orderForm.side.toLowerCase(),
        quantity: orderForm.quantity,
        order_type: orderForm.type.toLowerCase(),
        limit_price: orderForm.type === 'LIMIT' || orderForm.type === 'STOP_LIMIT' ? orderForm.price : null,
        stop_price: orderForm.type === 'STOP' || orderForm.type === 'STOP_LIMIT' ? orderForm.stopPrice : null,
        time_in_force: orderForm.timeInForce.toLowerCase(),
        portfolio_id: orderForm.portfolioId || 'default'
      };

      const response = await fetch('/api/v2/paper-trading/orders', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(orderData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to submit order');
      }

      const result = await response.json();
      
      // Add to local state immediately for UI responsiveness
      const newOrder: Order = {
        id: result.order_id,
        symbol: orderForm.symbol.toUpperCase(),
        side: orderForm.side,
        type: orderForm.type,
        quantity: orderForm.quantity,
        price: orderForm.price || undefined,
        stopPrice: orderForm.stopPrice || undefined,
        status: 'PENDING',
        timeInForce: orderForm.timeInForce,
        filledQuantity: 0,
        timestamp: new Date().toISOString(),
        portfolioId: orderForm.portfolioId || 'default'
      };
      
      setOrders(prev => [newOrder, ...prev]);
      
      // Reset form
      setOrderForm({
        symbol: 'AAPL',
        side: 'BUY',
        type: 'MARKET',
        quantity: 0,
        price: 0,
        stopPrice: 0,
        timeInForce: 'DAY',
        portfolioId: ''
      });

      // Show success message
      setError(null);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit order');
    } finally {
      setLoading(false);
    }
  };

  const cancelOrder = async (orderId: string) => {
    try {
      const response = await fetch(`/api/v2/paper-trading/orders/${orderId}/cancel`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to cancel order');
      }
      
      setOrders(prev => 
        prev.map(order => 
          order.id === orderId ? { ...order, status: 'CANCELLED' } : order
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel order');
    }
  };

  const toggleStrategy = async (strategyId: string, isActive: boolean) => {
    try {
      const response = await fetch(`/api/v2/trading/strategy/${strategyId}/${isActive ? 'start' : 'stop'}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to toggle strategy');
      
      setStrategies(prev => 
        prev.map(strategy => 
          strategy.id === strategyId ? { ...strategy, isActive } : strategy
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle strategy');
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatPercent = (percent: number) => {
    return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'FILLED': return 'text-green-600';
      case 'PENDING': return 'text-yellow-600';
      case 'CANCELLED': return 'text-gray-600';
      case 'REJECTED': return 'text-red-600';
      default: return 'text-blue-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'FILLED': return <CheckCircleIcon className="h-4 w-4" />;
      case 'PENDING': return <ClockIcon className="h-4 w-4" />;
      case 'CANCELLED': return <XCircleIcon className="h-4 w-4" />;
      case 'REJECTED': return <AlertTriangleIcon className="h-4 w-4" />;
      default: return <ClockIcon className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Trading Interface</h1>
          <p className="text-gray-600">Execute trades and manage positions</p>
        </div>
        <div className="flex items-center space-x-4">
          <Button
            variant={autoRefresh ? 'default' : 'outline'}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? <PauseIcon className="h-4 w-4 mr-2" /> : <PlayIcon className="h-4 w-4 mr-2" />}
            {autoRefresh ? 'Pause' : 'Resume'} Auto Refresh
          </Button>
          <Button variant="outline" onClick={() => {
            fetchOrders();
            fetchPositions();
            fetchMarketData();
          }}>
            <RefreshCwIcon className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangleIcon className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Market Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {Object.entries(marketData).map(([symbol, data]) => (
          <div key={symbol} className="cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedSymbol(symbol)}>
            <Card>
              <CardContent className="p-4">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-semibold">{symbol}</p>
                  <p className="text-2xl font-bold">{formatCurrency(data.price)}</p>
                </div>
                {data.changePercent >= 0 ? (
                  <TrendingUpIcon className="h-6 w-6 text-green-600" />
                ) : (
                  <TrendingDownIcon className="h-6 w-6 text-red-600" />
                )}
              </div>
              <div className={`text-sm ${
                data.changePercent >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {formatCurrency(data.change)} ({formatPercent(data.changePercent)})
              </div>
            </CardContent>
          </Card>
          </div>
        ))}
      </div>

      {/* Queue Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>Trade Execution Queue</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">{queueStats.pending}</div>
              <div className="text-sm text-gray-600">Pending Orders</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{queueStats.processing}</div>
              <div className="text-sm text-gray-600">Processing</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{queueStats.completed}</div>
              <div className="text-sm text-gray-600">Completed Today</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Order Entry */}
        <Card>
          <CardHeader>
            <CardTitle>Place Order</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="symbol">Symbol</Label>
                <Input
                  id="symbol"
                  value={orderForm.symbol}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, symbol: e.target.value.toUpperCase() }))}
                  placeholder="AAPL"
                />
              </div>
              <div>
                <Label htmlFor="side">Side</Label>
                <Select
                  value={orderForm.side}
                  onValueChange={(value: any) => setOrderForm(prev => ({ ...prev, side: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="BUY">Buy</SelectItem>
                    <SelectItem value="SELL">Sell</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="type">Order Type</Label>
                <Select
                  value={orderForm.type}
                  onValueChange={(value: any) => setOrderForm(prev => ({ ...prev, type: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MARKET">Market</SelectItem>
                    <SelectItem value="LIMIT">Limit</SelectItem>
                    <SelectItem value="STOP">Stop</SelectItem>
                    <SelectItem value="STOP_LIMIT">Stop Limit</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="timeInForce">Time in Force</Label>
                <Select
                  value={orderForm.timeInForce}
                  onValueChange={(value: any) => setOrderForm(prev => ({ ...prev, timeInForce: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="DAY">Day</SelectItem>
                    <SelectItem value="GTC">Good Till Cancelled</SelectItem>
                    <SelectItem value="IOC">Immediate or Cancel</SelectItem>
                    <SelectItem value="FOK">Fill or Kill</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label htmlFor="quantity">Quantity</Label>
              <Input
                id="quantity"
                type="number"
                value={orderForm.quantity}
                onChange={(e) => setOrderForm(prev => ({ ...prev, quantity: Number(e.target.value) }))}
              />
            </div>

            {(orderForm.type === 'LIMIT' || orderForm.type === 'STOP_LIMIT') && (
              <div>
                <Label htmlFor="price">Limit Price</Label>
                <Input
                  id="price"
                  type="number"
                  step="0.01"
                  value={orderForm.price}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, price: Number(e.target.value) }))}
                />
              </div>
            )}

            {(orderForm.type === 'STOP' || orderForm.type === 'STOP_LIMIT') && (
              <div>
                <Label htmlFor="stopPrice">Stop Price</Label>
                <Input
                  id="stopPrice"
                  type="number"
                  step="0.01"
                  value={orderForm.stopPrice}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, stopPrice: Number(e.target.value) }))}
                />
              </div>
            )}

            <Button 
              onClick={submitOrder} 
              disabled={loading || !orderForm.symbol || !orderForm.quantity}
              className="w-full"
            >
              {loading ? 'Submitting...' : `${orderForm.side} ${orderForm.symbol}`}
            </Button>
          </CardContent>
        </Card>

        {/* Positions */}
        <Card>
          <CardHeader>
            <CardTitle>Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {positions.map((position) => (
                <div key={position.symbol} className="p-3 border rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-semibold">{position.symbol}</p>
                      <p className="text-sm text-gray-600">
                        {position.quantity} shares @ {formatCurrency(position.averagePrice)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold">{formatCurrency(position.marketValue)}</p>
                      <p className={`text-sm ${
                        position.unrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatCurrency(position.unrealizedPnL)} ({formatPercent(position.unrealizedPnLPercent)})
                      </p>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    Day: {formatCurrency(position.dayChange)} ({formatPercent(position.dayChangePercent)})
                  </div>
                </div>
              ))}
              {positions.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <DollarSignIcon className="h-12 w-12 mx-auto mb-4" />
                  <p>No open positions</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Trading Strategies */}
        <Card>
          <CardHeader>
            <CardTitle>Trading Strategies</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {strategies.map((strategy) => (
                <div key={strategy.id} className="p-3 border rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-semibold">{strategy.name}</p>
                      <p className="text-sm text-gray-600">{strategy.description}</p>
                    </div>
                    <Button
                      size="sm"
                      variant={strategy.isActive ? 'destructive' : 'default'}
                      onClick={() => toggleStrategy(strategy.id, !strategy.isActive)}
                    >
                      {strategy.isActive ? 'Stop' : 'Start'}
                    </Button>
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500">Return:</span>
                      <span className={`ml-1 ${
                        strategy.performance.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatPercent(strategy.performance.totalReturn)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Win Rate:</span>
                      <span className="ml-1">{(strategy.performance.winRate * 100).toFixed(1)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Sharpe:</span>
                      <span className="ml-1">{strategy.performance.sharpeRatio.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Max DD:</span>
                      <span className="ml-1 text-red-600">{formatPercent(strategy.performance.maxDrawdown)}</span>
                    </div>
                  </div>
                  <Badge 
                    variant={strategy.isActive ? 'default' : 'secondary'}
                    className="mt-2"
                  >
                    {strategy.isActive ? 'Active' : 'Inactive'}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Orders Table */}
      <Card>
        <CardHeader>
          <CardTitle>Order History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {orders.map((order) => (
              <div key={order.id} className="flex justify-between items-center p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className={getStatusColor(order.status)}>
                    {getStatusIcon(order.status)}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={order.side === 'BUY' ? 'default' : 'secondary'}>
                        {order.side}
                      </Badge>
                      <span className="font-semibold">{order.symbol}</span>
                      <Badge variant="outline">{order.type}</Badge>
                    </div>
                    <p className="text-sm text-gray-600">
                      {order.quantity} shares
                      {order.price && ` @ ${formatCurrency(order.price)}`}
                      {order.filledQuantity > 0 && ` (${order.filledQuantity} filled)`}
                    </p>
                    <p className="text-xs text-gray-500">
                      {new Date(order.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <Badge className={getStatusColor(order.status)}>
                      {order.status}
                    </Badge>
                    {order.averagePrice && (
                      <p className="text-sm text-gray-600 mt-1">
                        Avg: {formatCurrency(order.averagePrice)}
                      </p>
                    )}
                  </div>
                  {order.status === 'PENDING' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => cancelOrder(order.id)}
                    >
                      Cancel
                    </Button>
                  )}
                </div>
              </div>
            ))}
            {orders.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <ClockIcon className="h-12 w-12 mx-auto mb-4" />
                <p>No orders found</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TradingInterface;