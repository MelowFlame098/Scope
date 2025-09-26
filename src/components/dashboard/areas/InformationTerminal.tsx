"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  NewspaperIcon,
  GlobeAltIcon,
  ChatBubbleLeftIcon,
  FireIcon,
  ClockIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  HeartIcon,
  ShareIcon
} from '@heroicons/react/24/outline';

interface InformationTerminalProps {
  user: any;
}

export const InformationTerminal: React.FC<InformationTerminalProps> = ({ user }) => {
  const [activeTab, setActiveTab] = useState('news');
  const [selectedCategory, setSelectedCategory] = useState('all');
  
  const [newsData, setNewsData] = useState([
    {
      id: 1,
      title: 'Federal Reserve Signals Potential Rate Cut in Q2 2024',
      summary: 'Fed officials hint at monetary policy shift amid cooling inflation data...',
      source: 'Reuters',
      category: 'economics',
      timestamp: '2 hours ago',
      sentiment: 'bullish',
      impact: 'high',
      tickers: ['SPY', 'QQQ', 'DXY'],
      url: '#'
    },
    {
      id: 2,
      title: 'Apple Reports Record Q4 Earnings, Beats Expectations',
      summary: 'iPhone sales surge drives revenue growth, services segment shows strong momentum...',
      source: 'Bloomberg',
      category: 'earnings',
      timestamp: '4 hours ago',
      sentiment: 'bullish',
      impact: 'medium',
      tickers: ['AAPL'],
      url: '#'
    },
    {
      id: 3,
      title: 'Tesla Announces New Gigafactory in Southeast Asia',
      summary: 'Expansion plans include battery production and vehicle assembly for regional markets...',
      source: 'CNBC',
      category: 'corporate',
      timestamp: '6 hours ago',
      sentiment: 'bullish',
      impact: 'medium',
      tickers: ['TSLA'],
      url: '#'
    },
    {
      id: 4,
      title: 'Cryptocurrency Market Sees Major Institutional Adoption',
      summary: 'Major banks announce crypto custody services, regulatory clarity improves...',
      source: 'CoinDesk',
      category: 'crypto',
      timestamp: '8 hours ago',
      sentiment: 'bullish',
      impact: 'high',
      tickers: ['BTC', 'ETH'],
      url: '#'
    }
  ]);

  const [socialFeed, setSocialFeed] = useState([
    {
      id: 1,
      author: '@TradingGuru',
      content: 'Massive volume spike in $NVDA pre-market. Something big is brewing 🚀',
      timestamp: '15m ago',
      likes: 234,
      shares: 45,
      sentiment: 'bullish',
      tickers: ['NVDA'],
      verified: true
    },
    {
      id: 2,
      author: '@MarketAnalyst',
      content: 'Fed meeting minutes suggest dovish stance. Bond yields dropping across the curve.',
      timestamp: '32m ago',
      likes: 156,
      shares: 78,
      sentiment: 'neutral',
      tickers: ['TLT', 'DXY'],
      verified: true
    },
    {
      id: 3,
      author: '@CryptoWhale',
      content: 'Bitcoin breaking key resistance at $45k. Next target $48k if volume sustains.',
      timestamp: '1h ago',
      likes: 892,
      shares: 234,
      sentiment: 'bullish',
      tickers: ['BTC'],
      verified: false
    },
    {
      id: 4,
      author: '@OptionsFlow',
      content: 'Unusual options activity in $TSLA. Large call volume at $250 strike expiring Friday.',
      timestamp: '2h ago',
      likes: 445,
      shares: 123,
      sentiment: 'bullish',
      tickers: ['TSLA'],
      verified: true
    }
  ]);

  const categories = [
    { id: 'all', name: 'All News', count: newsData.length },
    { id: 'economics', name: 'Economics', count: newsData.filter(n => n.category === 'economics').length },
    { id: 'earnings', name: 'Earnings', count: newsData.filter(n => n.category === 'earnings').length },
    { id: 'corporate', name: 'Corporate', count: newsData.filter(n => n.category === 'corporate').length },
    { id: 'crypto', name: 'Crypto', count: newsData.filter(n => n.category === 'crypto').length }
  ];

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'text-green-400 bg-green-400/10';
      case 'bearish': return 'text-red-400 bg-red-400/10';
      case 'neutral': return 'text-yellow-400 bg-yellow-400/10';
      default: return 'text-gray-400 bg-gray-400/10';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-600';
      case 'medium': return 'bg-orange-600';
      case 'low': return 'bg-green-600';
      default: return 'bg-gray-600';
    }
  };

  const filteredNews = selectedCategory === 'all' 
    ? newsData 
    : newsData.filter(news => news.category === selectedCategory);

  return (
    <div className="h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
        <TabsList className="grid w-full grid-cols-3 bg-gray-800">
          <TabsTrigger value="news" className="flex items-center space-x-2">
            <NewspaperIcon className="w-4 h-4" />
            <span>Global News</span>
          </TabsTrigger>
          <TabsTrigger value="social" className="flex items-center space-x-2">
            <ChatBubbleLeftIcon className="w-4 h-4" />
            <span>Social Feed</span>
          </TabsTrigger>
          <TabsTrigger value="trending" className="flex items-center space-x-2">
            <FireIcon className="w-4 h-4" />
            <span>Trending</span>
          </TabsTrigger>
        </TabsList>

        {/* Global News Tab */}
        <TabsContent value="news" className="flex-1 mt-4">
          <div className="space-y-4 h-full">
            {/* Category Filter */}
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <Button
                  key={category.id}
                  variant={selectedCategory === category.id ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedCategory(category.id)}
                  className="h-8"
                >
                  {category.name}
                  <Badge className="ml-2 h-4 px-1 text-xs">
                    {category.count}
                  </Badge>
                </Button>
              ))}
            </div>

            {/* News Feed */}
            <div className="space-y-3 overflow-y-auto flex-1">
              {filteredNews.map((news) => (
                <Card key={news.id} className="bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Badge className={getImpactColor(news.impact)}>
                          {news.impact} impact
                        </Badge>
                        <Badge className={getSentimentColor(news.sentiment)}>
                          {news.sentiment}
                        </Badge>
                      </div>
                      <div className="flex items-center text-gray-400 text-sm">
                        <ClockIcon className="w-4 h-4 mr-1" />
                        {news.timestamp}
                      </div>
                    </div>

                    <h3 className="text-white font-semibold mb-2 line-clamp-2">
                      {news.title}
                    </h3>
                    
                    <p className="text-gray-300 text-sm mb-3 line-clamp-2">
                      {news.summary}
                    </p>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-400 text-sm">{news.source}</span>
                        <div className="flex space-x-1">
                          {news.tickers.map((ticker) => (
                            <Badge key={ticker} variant="outline" className="text-xs">
                              ${ticker}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <Button variant="ghost" size="sm">
                        Read More
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Social Feed Tab */}
        <TabsContent value="social" className="flex-1 mt-4">
          <div className="space-y-3 overflow-y-auto h-full">
            {socialFeed.map((post) => (
              <Card key={post.id} className="bg-gray-800 border-gray-700">
                <CardContent className="p-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      {post.author.charAt(1).toUpperCase()}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="text-white font-medium">{post.author}</span>
                        {post.verified && (
                          <Badge className="bg-blue-600 text-white text-xs">
                            Verified
                          </Badge>
                        )}
                        <span className="text-gray-400 text-sm">{post.timestamp}</span>
                      </div>

                      <p className="text-gray-300 mb-3">{post.content}</p>

                      <div className="flex items-center justify-between">
                        <div className="flex space-x-1">
                          {post.tickers.map((ticker) => (
                            <Badge key={ticker} variant="outline" className="text-xs">
                              ${ticker}
                            </Badge>
                          ))}
                        </div>

                        <div className="flex items-center space-x-4 text-gray-400">
                          <button className="flex items-center space-x-1 hover:text-red-400 transition-colors">
                            <HeartIcon className="w-4 h-4" />
                            <span className="text-sm">{post.likes}</span>
                          </button>
                          <button className="flex items-center space-x-1 hover:text-blue-400 transition-colors">
                            <ShareIcon className="w-4 h-4" />
                            <span className="text-sm">{post.shares}</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Trending Tab */}
        <TabsContent value="trending" className="flex-1 mt-4">
          <div className="space-y-4">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <FireIcon className="w-5 h-5 mr-2 text-orange-500" />
                  Trending Topics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {[
                    { topic: 'Federal Reserve Rate Decision', mentions: 1234, trend: 'up' },
                    { topic: 'Apple Earnings Beat', mentions: 892, trend: 'up' },
                    { topic: 'Tesla Gigafactory', mentions: 567, trend: 'up' },
                    { topic: 'Bitcoin $45k Resistance', mentions: 445, trend: 'down' },
                    { topic: 'NVDA Options Flow', mentions: 334, trend: 'up' }
                  ].map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-700/50 rounded">
                      <div>
                        <div className="text-white font-medium">{item.topic}</div>
                        <div className="text-gray-400 text-sm">{item.mentions} mentions</div>
                      </div>
                      <div className={`flex items-center ${item.trend === 'up' ? 'text-green-400' : 'text-red-400'}`}>
                        {item.trend === 'up' ? (
                          <ArrowUpIcon className="w-4 h-4" />
                        ) : (
                          <ArrowDownIcon className="w-4 h-4" />
                        )}
                      </div>
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

export default InformationTerminal;