"use client";

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ChatBubbleLeftRightIcon,
  PaperAirplaneIcon,
  UserGroupIcon,
  LightBulbIcon,
  TrophyIcon,
  HeartIcon,
  ShareIcon,
  BookmarkIcon,
  EllipsisVerticalIcon
} from '@heroicons/react/24/outline';

interface ChatAreaProps {
  user: any;
}

interface Message {
  id: string;
  user: string;
  avatar: string;
  message: string;
  timestamp: Date;
  likes: number;
  replies: number;
  isTradeIdea?: boolean;
  symbol?: string;
  action?: 'BUY' | 'SELL';
  price?: number;
  target?: number;
  stopLoss?: number;
}

interface TradeIdea {
  id: string;
  user: string;
  avatar: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  currentPrice: number;
  targetPrice: number;
  stopLoss: number;
  reasoning: string;
  timestamp: Date;
  likes: number;
  comments: number;
  accuracy: number;
  followers: number;
}

export const ChatArea: React.FC<ChatAreaProps> = ({ user }) => {
  const [activeTab, setActiveTab] = useState('general');
  const [message, setMessage] = useState('');
  const [selectedRoom, setSelectedRoom] = useState('general');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      user: 'TradeMaster_Pro',
      avatar: 'TM',
      message: 'AAPL looking strong above $185. Volume is picking up nicely.',
      timestamp: new Date(Date.now() - 300000),
      likes: 12,
      replies: 3,
      isTradeIdea: true,
      symbol: 'AAPL',
      action: 'BUY',
      price: 185.50,
      target: 190.00,
      stopLoss: 182.00
    },
    {
      id: '2',
      user: 'CryptoKing',
      avatar: 'CK',
      message: 'Anyone else seeing this BTC pattern? Looks like a bull flag forming.',
      timestamp: new Date(Date.now() - 240000),
      likes: 8,
      replies: 5
    },
    {
      id: '3',
      user: 'OptionsGuru',
      avatar: 'OG',
      message: 'SPY puts are printing today. Volatility spike incoming.',
      timestamp: new Date(Date.now() - 180000),
      likes: 15,
      replies: 7
    },
    {
      id: '4',
      user: 'DayTrader_X',
      avatar: 'DX',
      message: 'TSLA breaking resistance at $240. Could see $250 soon.',
      timestamp: new Date(Date.now() - 120000),
      likes: 6,
      replies: 2,
      isTradeIdea: true,
      symbol: 'TSLA',
      action: 'BUY',
      price: 240.50,
      target: 250.00,
      stopLoss: 235.00
    }
  ]);

  const [tradeIdeas, setTradeIdeas] = useState<TradeIdea[]>([
    {
      id: '1',
      user: 'AlphaTrader',
      avatar: 'AT',
      symbol: 'NVDA',
      action: 'BUY',
      currentPrice: 875.50,
      targetPrice: 920.00,
      stopLoss: 850.00,
      reasoning: 'Strong earnings beat, AI momentum continues. Technical breakout above $870 resistance.',
      timestamp: new Date(Date.now() - 600000),
      likes: 24,
      comments: 12,
      accuracy: 78,
      followers: 1250
    },
    {
      id: '2',
      user: 'TechAnalyst',
      avatar: 'TA',
      symbol: 'MSFT',
      action: 'BUY',
      currentPrice: 415.25,
      targetPrice: 430.00,
      stopLoss: 405.00,
      reasoning: 'Cloud growth accelerating, Azure gaining market share. RSI oversold, good entry point.',
      timestamp: new Date(Date.now() - 900000),
      likes: 18,
      comments: 8,
      accuracy: 82,
      followers: 890
    },
    {
      id: '3',
      user: 'SwingMaster',
      avatar: 'SM',
      symbol: 'AMZN',
      action: 'SELL',
      currentPrice: 155.80,
      targetPrice: 145.00,
      stopLoss: 162.00,
      reasoning: 'Retail slowdown concerns, high valuation. Technical resistance at $156 level.',
      timestamp: new Date(Date.now() - 1200000),
      likes: 9,
      comments: 15,
      accuracy: 71,
      followers: 650
    }
  ]);

  const [leaderboard, setLeaderboard] = useState([
    { rank: 1, user: 'QuantKing', accuracy: 89, trades: 156, profit: '+$45,230' },
    { rank: 2, user: 'AlphaTrader', accuracy: 82, trades: 203, profit: '+$38,950' },
    { rank: 3, user: 'TechAnalyst', accuracy: 78, trades: 189, profit: '+$32,100' },
    { rank: 4, user: 'SwingMaster', accuracy: 75, trades: 145, profit: '+$28,750' },
    { rank: 5, user: 'OptionsGuru', accuracy: 73, trades: 167, profit: '+$25,400' }
  ]);

  const chatRooms = [
    { id: 'general', name: 'General', members: 1247, active: true },
    { id: 'stocks', name: 'Stocks', members: 892, active: true },
    { id: 'crypto', name: 'Crypto', members: 654, active: true },
    { id: 'options', name: 'Options', members: 423, active: true },
    { id: 'futures', name: 'Futures', members: 234, active: false }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (message.trim()) {
      const newMessage: Message = {
        id: Date.now().toString(),
        user: user?.name || 'You',
        avatar: user?.name?.substring(0, 2).toUpperCase() || 'YU',
        message: message.trim(),
        timestamp: new Date(),
        likes: 0,
        replies: 0
      };
      setMessages([...messages, newMessage]);
      setMessage('');
    }
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'now';
    if (minutes < 60) return `${minutes}m`;
    if (minutes < 1440) return `${Math.floor(minutes / 60)}h`;
    return `${Math.floor(minutes / 1440)}d`;
  };

  const getProfitColor = (profit: string) => {
    return profit.startsWith('+') ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div className="h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
        <TabsList className="grid w-full grid-cols-4 bg-gray-800">
          <TabsTrigger value="general">General Chat</TabsTrigger>
          <TabsTrigger value="ideas">Trade Ideas</TabsTrigger>
          <TabsTrigger value="rooms">Rooms</TabsTrigger>
          <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
        </TabsList>

        {/* General Chat Tab */}
        <TabsContent value="general" className="flex-1 mt-4">
          <Card className="bg-gray-800 border-gray-700 h-full flex flex-col">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-white flex items-center">
                  <ChatBubbleLeftRightIcon className="w-5 h-5 mr-2" />
                  General Chat
                </CardTitle>
                <Badge className="bg-green-600">
                  1,247 online
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col pt-0">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto space-y-3 mb-4">
                {messages.map((msg) => (
                  <div key={msg.id} className="flex space-x-3">
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-xs font-medium text-white">
                      {msg.avatar}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-white font-medium text-sm">{msg.user}</span>
                        <span className="text-gray-400 text-xs">{formatTime(msg.timestamp)}</span>
                        {msg.isTradeIdea && (
                          <Badge className="bg-yellow-600 text-xs">
                            <LightBulbIcon className="w-3 h-3 mr-1" />
                            Trade Idea
                          </Badge>
                        )}
                      </div>
                      <p className="text-gray-300 text-sm mb-2">{msg.message}</p>
                      {msg.isTradeIdea && (
                        <div className="bg-gray-700/50 rounded-lg p-3 mb-2">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-4">
                              <Badge className={msg.action === 'BUY' ? 'bg-green-600' : 'bg-red-600'}>
                                {msg.action} {msg.symbol}
                              </Badge>
                              <span className="text-white text-sm">Entry: ${msg.price}</span>
                              <span className="text-green-400 text-sm">Target: ${msg.target}</span>
                              <span className="text-red-400 text-sm">Stop: ${msg.stopLoss}</span>
                            </div>
                          </div>
                        </div>
                      )}
                      <div className="flex items-center space-x-4 text-xs text-gray-400">
                        <button className="flex items-center space-x-1 hover:text-red-400">
                          <HeartIcon className="w-4 h-4" />
                          <span>{msg.likes}</span>
                        </button>
                        <button className="flex items-center space-x-1 hover:text-blue-400">
                          <ChatBubbleLeftRightIcon className="w-4 h-4" />
                          <span>{msg.replies}</span>
                        </button>
                        <button className="hover:text-green-400">
                          <ShareIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* Message Input */}
              <div className="flex space-x-2">
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Type your message..."
                  className="flex-1 bg-gray-700 border-gray-600 text-white"
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <Button onClick={handleSendMessage} className="bg-blue-600 hover:bg-blue-700">
                  <PaperAirplaneIcon className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trade Ideas Tab */}
        <TabsContent value="ideas" className="flex-1 mt-4">
          <div className="space-y-4 h-full overflow-y-auto">
            {tradeIdeas.map((idea) => (
              <Card key={idea.id} className="bg-gray-800 border-gray-700">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center text-sm font-medium text-white">
                        {idea.avatar}
                      </div>
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-white font-medium">{idea.user}</span>
                          <Badge className="bg-blue-600 text-xs">
                            {idea.accuracy}% accuracy
                          </Badge>
                          <span className="text-gray-400 text-xs">
                            {idea.followers} followers
                          </span>
                        </div>
                        <span className="text-gray-400 text-xs">{formatTime(idea.timestamp)}</span>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm">
                      <EllipsisVerticalIcon className="w-4 h-4" />
                    </Button>
                  </div>

                  <div className="bg-gray-700/50 rounded-lg p-4 mb-3">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <Badge className={idea.action === 'BUY' ? 'bg-green-600' : 'bg-red-600'}>
                          {idea.action}
                        </Badge>
                        <span className="text-white font-bold text-lg">{idea.symbol}</span>
                        <span className="text-gray-300">${idea.currentPrice}</span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 mb-3">
                      <div className="text-center">
                        <div className="text-gray-400 text-xs">Target</div>
                        <div className="text-green-400 font-medium">${idea.targetPrice}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-400 text-xs">Stop Loss</div>
                        <div className="text-red-400 font-medium">${idea.stopLoss}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-gray-400 text-xs">R/R Ratio</div>
                        <div className="text-yellow-400 font-medium">
                          {((idea.targetPrice - idea.currentPrice) / (idea.currentPrice - idea.stopLoss)).toFixed(1)}:1
                        </div>
                      </div>
                    </div>

                    <p className="text-gray-300 text-sm">{idea.reasoning}</p>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-sm text-gray-400">
                      <button className="flex items-center space-x-1 hover:text-red-400">
                        <HeartIcon className="w-4 h-4" />
                        <span>{idea.likes}</span>
                      </button>
                      <button className="flex items-center space-x-1 hover:text-blue-400">
                        <ChatBubbleLeftRightIcon className="w-4 h-4" />
                        <span>{idea.comments}</span>
                      </button>
                      <button className="hover:text-green-400">
                        <ShareIcon className="w-4 h-4" />
                      </button>
                      <button className="hover:text-yellow-400">
                        <BookmarkIcon className="w-4 h-4" />
                      </button>
                    </div>
                    <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                      Follow Trade
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Rooms Tab */}
        <TabsContent value="rooms" className="flex-1 mt-4">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <UserGroupIcon className="w-5 h-5 mr-2" />
                Chat Rooms
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {chatRooms.map((room) => (
                  <div key={room.id} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${room.active ? 'bg-green-500' : 'bg-gray-500'}`} />
                      <div>
                        <div className="text-white font-medium">{room.name}</div>
                        <div className="text-gray-400 text-sm">{room.members} members</div>
                      </div>
                    </div>
                    <Button 
                      size="sm" 
                      variant={selectedRoom === room.id ? "default" : "outline"}
                      onClick={() => setSelectedRoom(room.id)}
                    >
                      {selectedRoom === room.id ? 'Active' : 'Join'}
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Leaderboard Tab */}
        <TabsContent value="leaderboard" className="flex-1 mt-4">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <TrophyIcon className="w-5 h-5 mr-2" />
                Top Traders
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {leaderboard.map((trader) => (
                  <div key={trader.rank} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                        trader.rank === 1 ? 'bg-yellow-500 text-black' :
                        trader.rank === 2 ? 'bg-gray-400 text-black' :
                        trader.rank === 3 ? 'bg-orange-600 text-white' :
                        'bg-gray-600 text-white'
                      }`}>
                        {trader.rank}
                      </div>
                      <div>
                        <div className="text-white font-medium">{trader.user}</div>
                        <div className="text-gray-400 text-sm">{trader.trades} trades</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-white font-medium">{trader.accuracy}%</div>
                      <div className={`text-sm font-medium ${getProfitColor(trader.profit)}`}>
                        {trader.profit}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ChatArea;