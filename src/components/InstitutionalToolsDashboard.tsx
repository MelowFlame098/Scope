'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import {
  BuildingOfficeIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ShieldCheckIcon,
  DocumentTextIcon,
  UserGroupIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  BanknotesIcon,
} from '@heroicons/react/24/outline';

interface InstitutionalTool {
  id: string;
  name: string;
  category: 'execution' | 'analytics' | 'risk' | 'compliance';
  description: string;
  status: 'active' | 'pending' | 'maintenance';
  usage: number;
  lastUsed: string;
}

interface OrderFlow {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  orderType: string;
  status: 'filled' | 'partial' | 'pending' | 'cancelled';
  timestamp: string;
  venue: string;
}

const institutionalTools: InstitutionalTool[] = [
  {
    id: '1',
    name: 'Algorithmic Execution',
    category: 'execution',
    description: 'Advanced order execution algorithms including TWAP, VWAP, and Implementation Shortfall',
    status: 'active',
    usage: 85,
    lastUsed: '2 hours ago',
  },
  {
    id: '2',
    name: 'Dark Pool Access',
    category: 'execution',
    description: 'Access to institutional dark pools for large block trading',
    status: 'active',
    usage: 62,
    lastUsed: '4 hours ago',
  },
  {
    id: '3',
    name: 'Portfolio Analytics',
    category: 'analytics',
    description: 'Comprehensive portfolio performance and attribution analysis',
    status: 'active',
    usage: 78,
    lastUsed: '1 hour ago',
  },
  {
    id: '4',
    name: 'Risk Management',
    category: 'risk',
    description: 'Real-time risk monitoring and position limits management',
    status: 'active',
    usage: 91,
    lastUsed: '30 minutes ago',
  },
  {
    id: '5',
    name: 'Compliance Monitoring',
    category: 'compliance',
    description: 'Automated compliance checking and regulatory reporting',
    status: 'maintenance',
    usage: 45,
    lastUsed: '1 day ago',
  },
  {
    id: '6',
    name: 'Market Impact Analysis',
    category: 'analytics',
    description: 'Pre-trade and post-trade market impact analysis',
    status: 'active',
    usage: 67,
    lastUsed: '3 hours ago',
  },
];

const orderFlow: OrderFlow[] = [
  {
    id: '1',
    symbol: 'AAPL',
    side: 'buy',
    quantity: 50000,
    price: 185.42,
    orderType: 'TWAP',
    status: 'filled',
    timestamp: '10:30 AM',
    venue: 'Dark Pool A',
  },
  {
    id: '2',
    symbol: 'MSFT',
    side: 'sell',
    quantity: 25000,
    price: 378.95,
    orderType: 'VWAP',
    status: 'partial',
    timestamp: '11:15 AM',
    venue: 'NYSE',
  },
  {
    id: '3',
    symbol: 'GOOGL',
    side: 'buy',
    quantity: 15000,
    price: 142.68,
    orderType: 'Implementation Shortfall',
    status: 'pending',
    timestamp: '11:45 AM',
    venue: 'NASDAQ',
  },
];

const InstitutionalToolsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'maintenance':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      case 'filled':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'partial':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
      case 'cancelled':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'execution':
        return <ChartBarIcon className="h-5 w-5" />;
      case 'analytics':
        return <DocumentTextIcon className="h-5 w-5" />;
      case 'risk':
        return <ShieldCheckIcon className="h-5 w-5" />;
      case 'compliance':
        return <ExclamationTriangleIcon className="h-5 w-5" />;
      default:
        return <BuildingOfficeIcon className="h-5 w-5" />;
    }
  };

  const filteredTools = selectedCategory === 'all' 
    ? institutionalTools 
    : institutionalTools.filter(tool => tool.category === selectedCategory);

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Institutional Tools
        </h2>
        <Button>
          <BuildingOfficeIcon className="h-4 w-4 mr-2" />
          Request Access
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="execution">Execution</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="compliance">Compliance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <CurrencyDollarIcon className="h-8 w-8 text-green-600" />
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Daily Volume</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">$2.4B</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <ChartBarIcon className="h-8 w-8 text-blue-600" />
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Active Orders</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">1,247</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <ShieldCheckIcon className="h-8 w-8 text-purple-600" />
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Risk Score</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">7.2/10</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <UserGroupIcon className="h-8 w-8 text-orange-600" />
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Active Users</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">156</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Tool Categories */}
          <div className="flex space-x-2 mb-4">
            <Button
              variant={selectedCategory === 'all' ? 'default' : 'outline'}
              onClick={() => setSelectedCategory('all')}
            >
              All Tools
            </Button>
            <Button
              variant={selectedCategory === 'execution' ? 'default' : 'outline'}
              onClick={() => setSelectedCategory('execution')}
            >
              Execution
            </Button>
            <Button
              variant={selectedCategory === 'analytics' ? 'default' : 'outline'}
              onClick={() => setSelectedCategory('analytics')}
            >
              Analytics
            </Button>
            <Button
              variant={selectedCategory === 'risk' ? 'default' : 'outline'}
              onClick={() => setSelectedCategory('risk')}
            >
              Risk
            </Button>
            <Button
              variant={selectedCategory === 'compliance' ? 'default' : 'outline'}
              onClick={() => setSelectedCategory('compliance')}
            >
              Compliance
            </Button>
          </div>

          {/* Tools Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {filteredTools.map((tool) => (
              <Card key={tool.id}>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                        {getCategoryIcon(tool.category)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {tool.name}
                        </h3>
                        <Badge className={getStatusColor(tool.status)}>
                          {tool.status}
                        </Badge>
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {tool.description}
                  </p>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500 dark:text-gray-400">Usage</span>
                      <span className="font-medium">{tool.usage}%</span>
                    </div>
                    <Progress value={tool.usage} className="h-2" />
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Last used: {tool.lastUsed}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="execution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Order Flow Management</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {orderFlow.map((order) => (
                  <div
                    key={order.id}
                    className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        {order.side === 'buy' ? (
                          <ArrowTrendingUpIcon className="h-5 w-5 text-green-600" />
                        ) : (
                          <ArrowTrendingDownIcon className="h-5 w-5 text-red-600" />
                        )}
                        <span className="font-semibold">{order.symbol}</span>
                      </div>
                      
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        <span className="capitalize">{order.side}</span> {formatNumber(order.quantity)} @ ${order.price}
                      </div>
                      
                      <Badge variant="outline">
                        {order.orderType}
                      </Badge>
                      
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {order.venue}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {order.timestamp}
                      </span>
                      <Badge className={getStatusColor(order.status)}>
                        {order.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Analytics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  Advanced Analytics Suite
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  Comprehensive portfolio performance and risk analytics
                </p>
                <Button>
                  <DocumentTextIcon className="h-4 w-4 mr-2" />
                  Generate Report
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Dashboard</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <CheckCircleIcon className="h-6 w-6 text-green-600" />
                    <div>
                      <h4 className="font-semibold text-green-800 dark:text-green-400">
                        Regulatory Compliance
                      </h4>
                      <p className="text-sm text-green-600 dark:text-green-500">
                        All systems compliant
                      </p>
                    </div>
                  </div>
                  <Badge className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                    Active
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <ClockIcon className="h-6 w-6 text-yellow-600" />
                    <div>
                      <h4 className="font-semibold text-yellow-800 dark:text-yellow-400">
                        Position Limits
                      </h4>
                      <p className="text-sm text-yellow-600 dark:text-yellow-500">
                        Monitoring active positions
                      </p>
                    </div>
                  </div>
                  <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400">
                    Monitoring
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default InstitutionalToolsDashboard;