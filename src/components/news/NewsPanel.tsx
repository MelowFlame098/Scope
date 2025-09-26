"use client";

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  NewspaperIcon,
  ClockIcon,
  ArrowTopRightOnSquareIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  publishedAt: string;
  url: string;
  category: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  relevanceScore: number;
}

const NewsPanel: React.FC = () => {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // Mock news data for demonstration
  useEffect(() => {
    const loadNews = async () => {
      setIsLoading(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const mockNews: NewsItem[] = [
        {
          id: '1',
          title: 'Federal Reserve Signals Potential Rate Cut in Q2 2024',
          summary: 'Fed officials hint at possible monetary policy adjustments amid cooling inflation data and economic indicators.',
          source: 'Reuters',
          publishedAt: '2024-01-15T10:30:00Z',
          url: '#',
          category: 'monetary-policy',
          sentiment: 'positive',
          relevanceScore: 0.95
        },
        {
          id: '2',
          title: 'Tech Stocks Rally on Strong Q4 Earnings Reports',
          summary: 'Major technology companies exceed expectations, driving sector-wide gains in pre-market trading.',
          source: 'Bloomberg',
          publishedAt: '2024-01-15T09:15:00Z',
          url: '#',
          category: 'earnings',
          sentiment: 'positive',
          relevanceScore: 0.88
        },
        {
          id: '3',
          title: 'Oil Prices Surge on Middle East Tensions',
          summary: 'Crude oil futures jump 3% as geopolitical concerns affect global supply chain expectations.',
          source: 'CNBC',
          publishedAt: '2024-01-15T08:45:00Z',
          url: '#',
          category: 'commodities',
          sentiment: 'negative',
          relevanceScore: 0.82
        },
        {
          id: '4',
          title: 'Bitcoin Reaches New Monthly High Amid ETF Optimism',
          summary: 'Cryptocurrency markets gain momentum following positive regulatory developments and institutional adoption.',
          source: 'CoinDesk',
          publishedAt: '2024-01-15T07:20:00Z',
          url: '#',
          category: 'crypto',
          sentiment: 'positive',
          relevanceScore: 0.76
        },
        {
          id: '5',
          title: 'European Markets Open Higher on ECB Policy Hopes',
          summary: 'European indices gain as investors anticipate potential policy shifts from the European Central Bank.',
          source: 'Financial Times',
          publishedAt: '2024-01-15T06:00:00Z',
          url: '#',
          category: 'markets',
          sentiment: 'positive',
          relevanceScore: 0.71
        },
        {
          id: '6',
          title: 'Housing Market Shows Signs of Stabilization',
          summary: 'New data suggests the real estate market may be finding its footing after months of volatility.',
          source: 'Wall Street Journal',
          publishedAt: '2024-01-14T16:30:00Z',
          url: '#',
          category: 'real-estate',
          sentiment: 'neutral',
          relevanceScore: 0.68
        }
      ];
      
      setNews(mockNews);
      setIsLoading(false);
    };
    
    loadNews();
  }, []);

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) {
      return 'Just now';
    } else if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    } else {
      const diffInDays = Math.floor(diffInHours / 24);
      return `${diffInDays}d ago`;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'negative':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getCategoryColor = (category: string) => {
    const colors: { [key: string]: string } = {
      'monetary-policy': 'bg-blue-500/20 text-blue-400',
      'earnings': 'bg-purple-500/20 text-purple-400',
      'commodities': 'bg-yellow-500/20 text-yellow-400',
      'crypto': 'bg-orange-500/20 text-orange-400',
      'markets': 'bg-green-500/20 text-green-400',
      'real-estate': 'bg-pink-500/20 text-pink-400',
    };
    return colors[category] || 'bg-gray-500/20 text-gray-400';
  };

  const filteredNews = selectedCategory === 'all' 
    ? news 
    : news.filter(item => item.category === selectedCategory);

  const categories = [
    { id: 'all', label: 'All News' },
    { id: 'monetary-policy', label: 'Fed Policy' },
    { id: 'earnings', label: 'Earnings' },
    { id: 'markets', label: 'Markets' },
    { id: 'crypto', label: 'Crypto' },
    { id: 'commodities', label: 'Commodities' },
  ];

  if (isLoading) {
    return (
      <Card className="bg-gray-800 border-gray-700 h-full">
        <CardHeader>
          <CardTitle className="text-white flex items-center space-x-2">
            <NewspaperIcon className="h-5 w-5" />
            <span>Market News</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-600 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-600 rounded w-1/2 mb-2"></div>
                <div className="h-3 bg-gray-600 rounded w-1/4"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gray-800 border-gray-700 h-full">
      <CardHeader>
        <CardTitle className="text-white flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <NewspaperIcon className="h-5 w-5" />
            <span>Market News</span>
          </div>
          <Badge variant="secondary" className="bg-blue-500/20 text-blue-400">
            {filteredNews.length}
          </Badge>
        </CardTitle>
        
        {/* Category Filter */}
        <div className="flex flex-wrap gap-2 mt-4">
          {categories.map((category) => (
            <Button
              key={category.id}
              variant={selectedCategory === category.id ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedCategory(category.id)}
              className={`text-xs ${
                selectedCategory === category.id
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-transparent text-gray-400 border-gray-600 hover:bg-gray-700'
              }`}
            >
              {category.label}
            </Button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4 max-h-96 overflow-y-auto">
        {filteredNews.length === 0 ? (
          <div className="text-center py-8">
            <NewspaperIcon className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No news available for this category</p>
          </div>
        ) : (
          filteredNews.map((item) => (
            <div
              key={item.id}
              className="p-4 bg-gray-700/50 rounded-lg border border-gray-600/50 hover:bg-gray-700 transition-colors cursor-pointer group"
              onClick={() => window.open(item.url, '_blank')}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Badge className={getCategoryColor(item.category)}>
                    {item.category.replace('-', ' ')}
                  </Badge>
                  <Badge className={getSentimentColor(item.sentiment)}>
                    {item.sentiment}
                  </Badge>
                </div>
                <ArrowTopRightOnSquareIcon className="h-4 w-4 text-gray-400 group-hover:text-white transition-colors" />
              </div>
              
              <h3 className="text-white font-medium text-sm mb-2 line-clamp-2 group-hover:text-blue-400 transition-colors">
                {item.title}
              </h3>
              
              <p className="text-gray-400 text-xs mb-3 line-clamp-2">
                {item.summary}
              </p>
              
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-3">
                  <span className="text-gray-500">{item.source}</span>
                  <div className="flex items-center space-x-1 text-gray-500">
                    <ClockIcon className="h-3 w-3" />
                    <span>{formatTimeAgo(item.publishedAt)}</span>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span className="text-gray-500">{Math.round(item.relevanceScore * 100)}%</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
        
        {filteredNews.length > 0 && (
          <div className="pt-4 border-t border-gray-600">
            <Button
              variant="ghost"
              className="w-full text-blue-400 hover:text-blue-300 hover:bg-blue-500/10"
              onClick={() => console.log('View all news')}
            >
              View All News
              <ChevronRightIcon className="h-4 w-4 ml-2" />
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export { NewsPanel };