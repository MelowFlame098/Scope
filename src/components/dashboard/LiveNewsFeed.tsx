'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  NewspaperIcon, 
  TrendingUp as TrendingUpIcon, 
  TrendingDownIcon, 
  ClockIcon,
  ExternalLinkIcon,
  SearchIcon,
  FilterIcon,
  RefreshCwIcon,
  AlertTriangleIcon,
  BarChart3Icon
} from 'lucide-react';
import { newsService, NewsArticle, MarketNews, NewsFilter } from '@/services/NewsService';

interface LiveNewsFeedProps {
  selectedSymbol?: string;
  className?: string;
}

const LiveNewsFeed: React.FC<LiveNewsFeedProps> = ({ selectedSymbol, className = '' }) => {
  const [news, setNews] = useState<MarketNews | null>(null);
  const [assetNews, setAssetNews] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<NewsArticle[]>([]);
  const [activeTab, setActiveTab] = useState('breaking');
  const [filter, setFilter] = useState<NewsFilter>({});
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Load news data
  const loadNews = useCallback(async () => {
    try {
      setLoading(true);
      const [marketNews, symbolNews] = await Promise.all([
        newsService.getNews(filter),
        selectedSymbol ? newsService.getAssetNews(selectedSymbol, 20) : Promise.resolve([])
      ]);
      
      setNews(marketNews);
      setAssetNews(symbolNews);
    } catch (error) {
      console.error('Failed to load news:', error);
    } finally {
      setLoading(false);
    }
  }, [filter, selectedSymbol]);

  // Search news
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      const results = await newsService.searchNews(query);
      setSearchResults(results);
    } catch (error) {
      console.error('Failed to search news:', error);
    }
  }, []);

  // Refresh news
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await loadNews();
    setTimeout(() => setIsRefreshing(false), 1000);
  }, [loadNews]);

  // Initialize and set up real-time updates
  useEffect(() => {
    loadNews();

    // Subscribe to real-time updates
    const handleNewArticle = (article: NewsArticle) => {
      if (selectedSymbol && article.symbols.includes(selectedSymbol)) {
        setAssetNews(prev => [article, ...prev.slice(0, 19)]);
      }
      
      setNews(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          topStories: [article, ...prev.topStories.slice(0, 9)]
        };
      });
    };

    const handleBreakingNews = (article: NewsArticle) => {
      setNews(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          breakingNews: [article, ...prev.breakingNews.slice(0, 2)]
        };
      });
    };

    newsService.on('newArticle', handleNewArticle);
    newsService.on('breakingNews', handleBreakingNews);

    // Subscribe to symbol-specific updates
    if (selectedSymbol) {
      newsService.subscribeToSymbol(selectedSymbol);
    }

    return () => {
      newsService.off('newArticle', handleNewArticle);
      newsService.off('breakingNews', handleBreakingNews);
      if (selectedSymbol) {
        newsService.unsubscribeFromSymbol(selectedSymbol);
      }
    };
  }, [selectedSymbol, loadNews]);

  // Handle search input
  useEffect(() => {
    const debounceTimer = setTimeout(() => {
      handleSearch(searchQuery);
    }, 300);

    return () => clearTimeout(debounceTimer);
  }, [searchQuery, handleSearch]);

  const getSentimentColor = (sentiment: NewsArticle['sentiment']) => {
    switch (sentiment) {
      case 'bullish': return 'text-green-600 bg-green-50';
      case 'bearish': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getImpactColor = (impact: NewsArticle['impact']) => {
    switch (impact) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  const renderNewsArticle = (article: NewsArticle, showSymbols = true) => (
    <div key={article.id} className="border-b border-gray-200 pb-4 mb-4 last:border-b-0">
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 mb-1 line-clamp-2">
            {article.title}
          </h3>
          <p className="text-sm text-gray-600 mb-2 line-clamp-2">
            {article.summary}
          </p>
        </div>
        {article.imageUrl && (
          <img 
            src={article.imageUrl} 
            alt={article.title}
            className="w-16 h-16 object-cover rounded-lg ml-3 flex-shrink-0"
          />
        )}
      </div>
      
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">{article.source}</span>
          <span className="text-xs text-gray-400">•</span>
          <span className="text-xs text-gray-500 flex items-center">
            <ClockIcon className="w-3 h-3 mr-1" />
            {formatTimeAgo(article.publishedAt)}
          </span>
          <span className="text-xs text-gray-400">•</span>
          <span className="text-xs text-gray-500">{article.readTime}min read</span>
        </div>
        
        <div className="flex items-center space-x-1">
          <Badge className={`text-xs ${getSentimentColor(article.sentiment)}`}>
            {article.sentiment === 'bullish' && <TrendingUpIcon className="w-3 h-3 mr-1" />}
            {article.sentiment === 'bearish' && <TrendingDownIcon className="w-3 h-3 mr-1" />}
            {article.sentiment.toUpperCase()}
          </Badge>
          <Badge className={`text-xs ${getImpactColor(article.impact)}`}>
            {article.impact.toUpperCase()}
          </Badge>
        </div>
      </div>
      
      {showSymbols && article.symbols.length > 0 && (
        <div className="flex items-center space-x-1 mt-2">
          {article.symbols.map(symbol => (
            <Badge key={symbol} variant="outline" className="text-xs">
              {symbol}
            </Badge>
          ))}
        </div>
      )}
      
      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center space-x-2">
          {article.tags.slice(0, 3).map(tag => (
            <span key={tag} className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
              #{tag}
            </span>
          ))}
        </div>
        
        <Button variant="ghost" size="sm" className="text-xs">
          <ExternalLinkIcon className="w-3 h-3 mr-1" />
          Read More
        </Button>
      </div>
    </div>
  );

  const renderSentimentOverview = (articles: NewsArticle[]) => {
    const sentiment = newsService.getSentimentAnalysis(articles);
    
    return (
      <div className="bg-gray-50 p-4 rounded-lg mb-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-gray-900">Market Sentiment</h4>
          <Badge className={getSentimentColor(sentiment.overall)}>
            {sentiment.overall.toUpperCase()}
          </Badge>
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-green-600">
              {sentiment.distribution.bullish}
            </div>
            <div className="text-xs text-gray-600">Bullish</div>
          </div>
          <div>
            <div className="text-lg font-bold text-gray-600">
              {sentiment.distribution.neutral}
            </div>
            <div className="text-xs text-gray-600">Neutral</div>
          </div>
          <div>
            <div className="text-lg font-bold text-red-600">
              {sentiment.distribution.bearish}
            </div>
            <div className="text-xs text-gray-600">Bearish</div>
          </div>
        </div>
        
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
            <span>Sentiment Score</span>
            <span>{sentiment.score.toFixed(2)}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${
                sentiment.score > 0 ? 'bg-green-500' : 'bg-red-500'
              }`}
              style={{ 
                width: `${Math.abs(sentiment.score) * 50 + 50}%`,
                marginLeft: sentiment.score < 0 ? `${(1 + sentiment.score) * 50}%` : '0'
              }}
            />
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center">
            <NewspaperIcon className="w-5 h-5 mr-2" />
            Live News Feed
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-gray-600">Loading news...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <NewspaperIcon className="w-5 h-5 mr-2" />
            Live News Feed
            {selectedSymbol && (
              <Badge variant="outline" className="ml-2">
                {selectedSymbol}
              </Badge>
            )}
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <RefreshCwIcon className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="outline" size="sm">
              <FilterIcon className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        <div className="relative">
          <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search news..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
      </CardHeader>

      <CardContent>
        {searchQuery && searchResults.length > 0 ? (
          <div>
            <h3 className="font-semibold mb-4">Search Results ({searchResults.length})</h3>
            <div className="max-h-96 overflow-y-auto">
              {searchResults.map(article => renderNewsArticle(article))}
            </div>
          </div>
        ) : (
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="breaking">Breaking</TabsTrigger>
              <TabsTrigger value="top">Top Stories</TabsTrigger>
              <TabsTrigger value="asset">
                {selectedSymbol || 'Asset'}
              </TabsTrigger>
              <TabsTrigger value="sectors">Sectors</TabsTrigger>
            </TabsList>

            <TabsContent value="breaking" className="mt-4">
              {news?.breakingNews && news.breakingNews.length > 0 ? (
                <div>
                  <div className="flex items-center mb-4">
                    <AlertTriangleIcon className="w-5 h-5 text-red-500 mr-2" />
                    <span className="font-semibold text-red-600">Breaking News</span>
                  </div>
                  <div className="max-h-96 overflow-y-auto">
                    {news.breakingNews.map(article => renderNewsArticle(article))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No breaking news at the moment
                </div>
              )}
            </TabsContent>

            <TabsContent value="top" className="mt-4">
              {news?.topStories && (
                <div>
                  {renderSentimentOverview(news.topStories)}
                  <div className="max-h-96 overflow-y-auto">
                    {news.topStories.map(article => renderNewsArticle(article))}
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="asset" className="mt-4">
              {selectedSymbol ? (
                assetNews.length > 0 ? (
                  <div>
                    {renderSentimentOverview(assetNews)}
                    <div className="max-h-96 overflow-y-auto">
                      {assetNews.map(article => renderNewsArticle(article, false))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No news available for {selectedSymbol}
                  </div>
                )
              ) : (
                <div className="text-center py-8 text-gray-500">
                  Select an asset to view specific news
                </div>
              )}
            </TabsContent>

            <TabsContent value="sectors" className="mt-4">
              <Tabs defaultValue="earnings" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="earnings">Earnings</TabsTrigger>
                  <TabsTrigger value="regulatory">Regulatory</TabsTrigger>
                  <TabsTrigger value="crypto">Crypto</TabsTrigger>
                </TabsList>

                <TabsContent value="earnings" className="mt-4">
                  <div className="max-h-96 overflow-y-auto">
                    {news?.earnings?.map(article => renderNewsArticle(article)) || (
                      <div className="text-center py-8 text-gray-500">
                        No earnings news available
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="regulatory" className="mt-4">
                  <div className="max-h-96 overflow-y-auto">
                    {news?.regulatory?.map(article => renderNewsArticle(article)) || (
                      <div className="text-center py-8 text-gray-500">
                        No regulatory news available
                      </div>
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="crypto" className="mt-4">
                  <div className="max-h-96 overflow-y-auto">
                    {news?.crypto?.map(article => renderNewsArticle(article)) || (
                      <div className="text-center py-8 text-gray-500">
                        No crypto news available
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </TabsContent>
          </Tabs>
        )}
      </CardContent>
    </Card>
  );
};

export default LiveNewsFeed;