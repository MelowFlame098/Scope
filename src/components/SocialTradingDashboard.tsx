'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import {
  UserGroupIcon,
  StarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  EyeIcon,
  HeartIcon,
  ChatBubbleLeftIcon,
  ShareIcon,
  ClockIcon,
  CurrencyDollarIcon,
} from '@heroicons/react/24/outline';
import { StarIcon as StarIconSolid } from '@heroicons/react/24/solid';

interface Trader {
  id: string;
  name: string;
  username: string;
  avatar?: string;
  verified: boolean;
  followers: number;
  following: number;
  totalReturn: number;
  monthlyReturn: number;
  winRate: number;
  riskScore: number;
  copiers: number;
  isFollowing: boolean;
}

interface Trade {
  id: string;
  trader: Trader;
  symbol: string;
  action: 'buy' | 'sell';
  amount: number;
  price: number;
  timestamp: string;
  pnl?: number;
  likes: number;
  comments: number;
  isLiked: boolean;
}

const topTraders: Trader[] = [
  {
    id: '1',
    name: 'Alex Chen',
    username: '@alextrader',
    verified: true,
    followers: 15420,
    following: 234,
    totalReturn: 156.8,
    monthlyReturn: 12.4,
    winRate: 78.5,
    riskScore: 6.2,
    copiers: 892,
    isFollowing: false,
  },
  {
    id: '2',
    name: 'Sarah Johnson',
    username: '@sarahfx',
    verified: true,
    followers: 12890,
    following: 189,
    totalReturn: 134.2,
    monthlyReturn: 9.8,
    winRate: 82.1,
    riskScore: 4.8,
    copiers: 654,
    isFollowing: true,
  },
  {
    id: '3',
    name: 'Mike Rodriguez',
    username: '@miketrading',
    verified: false,
    followers: 8765,
    following: 156,
    totalReturn: 98.7,
    monthlyReturn: 15.2,
    winRate: 71.3,
    riskScore: 7.9,
    copiers: 423,
    isFollowing: false,
  },
];

const recentTrades: Trade[] = [
  {
    id: '1',
    trader: topTraders[0],
    symbol: 'AAPL',
    action: 'buy',
    amount: 100,
    price: 185.42,
    timestamp: '2 hours ago',
    pnl: 234.50,
    likes: 45,
    comments: 12,
    isLiked: false,
  },
  {
    id: '2',
    trader: topTraders[1],
    symbol: 'TSLA',
    action: 'sell',
    amount: 50,
    price: 242.18,
    timestamp: '4 hours ago',
    pnl: -156.30,
    likes: 23,
    comments: 8,
    isLiked: true,
  },
  {
    id: '3',
    trader: topTraders[2],
    symbol: 'NVDA',
    action: 'buy',
    amount: 25,
    price: 456.78,
    timestamp: '6 hours ago',
    pnl: 89.25,
    likes: 67,
    comments: 15,
    isLiked: false,
  },
];

const SocialTradingDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('feed');
  const [traders, setTraders] = useState(topTraders);
  const [trades, setTrades] = useState(recentTrades);

  const toggleFollow = (traderId: string) => {
    setTraders(prev => prev.map(trader => 
      trader.id === traderId 
        ? { ...trader, isFollowing: !trader.isFollowing }
        : trader
    ));
  };

  const toggleLike = (tradeId: string) => {
    setTrades(prev => prev.map(trade => 
      trade.id === tradeId 
        ? { 
            ...trade, 
            isLiked: !trade.isLiked,
            likes: trade.isLiked ? trade.likes - 1 : trade.likes + 1
          }
        : trade
    ));
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const getRiskColor = (score: number) => {
    if (score <= 3) return 'text-green-600';
    if (score <= 6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Social Trading
        </h2>
        <Button>
          <UserGroupIcon className="h-4 w-4 mr-2" />
          Find Traders
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="feed">Trading Feed</TabsTrigger>
          <TabsTrigger value="traders">Top Traders</TabsTrigger>
          <TabsTrigger value="portfolio">My Copies</TabsTrigger>
        </TabsList>

        <TabsContent value="feed" className="space-y-4">
          <div className="space-y-4">
            {trades.map((trade) => (
              <Card key={trade.id}>
                <CardContent className="p-4">
                  <div className="flex items-start space-x-3">
                    <Avatar>
                      <AvatarImage src={trade.trader.avatar || ''} alt={trade.trader.name} />
                      <AvatarFallback>
                        {trade.trader.name.split(' ').map(n => n[0]).join('')}
                      </AvatarFallback>
                    </Avatar>
                    
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="font-semibold text-gray-900 dark:text-white">
                          {trade.trader.name}
                        </span>
                        <span className="text-gray-500 dark:text-gray-400">
                          {trade.trader.username}
                        </span>
                        {trade.trader.verified && (
                          <StarIconSolid className="h-4 w-4 text-blue-500" />
                        )}
                        <span className="text-sm text-gray-500 dark:text-gray-400">
                          {trade.timestamp}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-4 mb-3">
                        <Badge 
                          variant={trade.action === 'buy' ? 'default' : 'destructive'}
                          className="uppercase"
                        >
                          {trade.action}
                        </Badge>
                        <span className="font-semibold text-lg">
                          {trade.symbol}
                        </span>
                        <span className="text-gray-600 dark:text-gray-400">
                          {trade.amount} shares @ ${trade.price}
                        </span>
                      </div>
                      
                      {trade.pnl && (
                        <div className="flex items-center space-x-2 mb-3">
                          <span className="text-sm text-gray-500">P&L:</span>
                          <span className={`font-semibold ${
                            trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                          </span>
                          {trade.pnl >= 0 ? (
                            <ArrowTrendingUpIcon className="h-4 w-4 text-green-600" />
              ) : (
                <ArrowTrendingDownIcon className="h-4 w-4 text-red-600" />
                          )}
                        </div>
                      )}
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                        <button
                          onClick={() => toggleLike(trade.id)}
                          className={`flex items-center space-x-1 hover:text-red-500 ${
                            trade.isLiked ? 'text-red-500' : ''
                          }`}
                        >
                          <HeartIcon className="h-4 w-4" />
                          <span>{trade.likes}</span>
                        </button>
                        <button className="flex items-center space-x-1 hover:text-blue-500">
                          <ChatBubbleLeftIcon className="h-4 w-4" />
                          <span>{trade.comments}</span>
                        </button>
                        <button className="flex items-center space-x-1 hover:text-green-500">
                          <ShareIcon className="h-4 w-4" />
                          <span>Share</span>
                        </button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="traders" className="space-y-4">
          <div className="grid gap-4">
            {traders.map((trader) => (
              <Card key={trader.id}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4">
                      <Avatar className="h-12 w-12">
                        <AvatarImage src={trader.avatar || ''} alt={trader.name} />
                        <AvatarFallback>
                          {trader.name.split(' ').map(n => n[0]).join('')}
                        </AvatarFallback>
                      </Avatar>
                      
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="font-semibold text-gray-900 dark:text-white">
                            {trader.name}
                          </h3>
                          {trader.verified && (
                            <StarIconSolid className="h-4 w-4 text-blue-500" />
                          )}
                        </div>
                        <p className="text-gray-500 dark:text-gray-400 mb-2">
                          {trader.username}
                        </p>
                        
                        <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                          <span>{formatNumber(trader.followers)} followers</span>
                          <span>{formatNumber(trader.copiers)} copiers</span>
                        </div>
                      </div>
                    </div>
                    
                    <Button
                      variant={trader.isFollowing ? "outline" : "default"}
                      onClick={() => toggleFollow(trader.id)}
                    >
                      {trader.isFollowing ? 'Following' : 'Follow'}
                    </Button>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Total Return</p>
                      <p className="font-semibold text-green-600">
                        +{trader.totalReturn}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Monthly</p>
                      <p className="font-semibold text-green-600">
                        +{trader.monthlyReturn}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Win Rate</p>
                      <p className="font-semibold text-gray-900 dark:text-white">
                        {trader.winRate}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Risk Score</p>
                      <p className={`font-semibold ${getRiskColor(trader.riskScore)}`}>
                        {trader.riskScore}/10
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="portfolio" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Copy Trading Portfolio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <UserGroupIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  No Active Copies
                </h3>
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  Start copying successful traders to build your portfolio
                </p>
                <Button>
                  <EyeIcon className="h-4 w-4 mr-2" />
                  Browse Traders
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SocialTradingDashboard;