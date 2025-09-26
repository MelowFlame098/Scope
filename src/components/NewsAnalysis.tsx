'use client';

import React, { useEffect, useState } from 'react';
import {
  NewspaperIcon,
  ClockIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  EyeIcon,
  ShareIcon,
  HeartIcon,
  ChatBubbleLeftIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  BookmarkIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { HeartIcon as HeartIconSolid, BookmarkIcon as BookmarkIconSolid } from '@heroicons/react/24/solid';
import { useStore } from '../store/useStore';
import { useNewsUpdates } from '../hooks';
import { formatRelativeTime, getSentimentColor } from '../utils';
import LoadingSpinner from './ui/LoadingSpinner';

const NewsAnalysis: React.FC = () => {
  const {
    news,
    selectedAssets,
    searchQuery,
    isLoading,
    setSearchQuery,
    fetchNews,
    addNotification,
  } = useStore();
  
  // Subscribe to real-time news updates
  useNewsUpdates();
  
  const [sentimentFilter, setSentimentFilter] = React.useState<'all' | 'positive' | 'negative' | 'neutral'>('all');
  const [sortBy, setSortBy] = React.useState<'publishedAt' | 'relevance' | 'sentiment'>('publishedAt');
  const [selectedNewsItem, setSelectedNewsItem] = React.useState<any>(null);
  const [sourceFilter, setSourceFilter] = React.useState<string>('all');
  const [isClient, setIsClient] = useState(false);

  // Set client-side flag to prevent hydration issues
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Fetch news on component mount
  useEffect(() => {
    fetchNews();
  }, [fetchNews]);

  const filteredNews = React.useMemo(() => {
    let filtered = news.filter(item => {
      const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           item.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           item.relevantAssets?.some((asset: string) => asset.toLowerCase().includes(searchQuery.toLowerCase()));
      
      const matchesSentiment = sentimentFilter === 'all' || item.sentiment === sentimentFilter;
      
      const matchesSource = sourceFilter === 'all' || item.source === sourceFilter;
      
      const matchesAssets = selectedAssets.length === 0 ||
                           selectedAssets.some(asset =>
                             item.relevantAssets?.includes(asset.symbol) ||
                             item.relevantAssets?.includes(asset.id)
                           );
      
      return matchesSearch && matchesSentiment && matchesSource && matchesAssets;
    });

    // Sort news
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'publishedAt':
          return new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime();
        case 'relevance':
          // Since relevanceScore doesn't exist, sort by publishedAt as fallback
          return new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime();
        case 'sentiment':
          // Since sentimentScore doesn't exist, sort by sentiment alphabetically
          return a.sentiment.localeCompare(b.sentiment);
        default:
          return 0;
      }
    });

    return filtered;
  }, [news, searchQuery, sentimentFilter, sourceFilter, sortBy, selectedAssets]);

  const uniqueSources = React.useMemo(() => {
    const sources = Array.from(new Set(news.map(item => item.source)));
    return sources.sort();
  }, [news]);

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <ArrowUpIcon className="h-4 w-4 text-green-500" />;
  case 'negative':
    return <ArrowDownIcon className="h-4 w-4 text-red-500" />;
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full"></div>;
    }
  };

  const handleLike = async (newsId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      // TODO: Implement like functionality when backend API is available
      addNotification({
        id: Date.now().toString(),
        type: 'info',
        message: 'Like functionality coming soon',
        timestamp: new Date().toISOString(),
        read: false,
      });
    } catch (error) {
      addNotification({
        id: Date.now().toString(),
        type: 'error',
        message: 'Failed to like news item',
        timestamp: new Date().toISOString(),
        read: false,
      });
    }
  };

  const handleBookmark = async (newsId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      // TODO: Implement bookmark functionality when backend API is available
      const newsItem = news.find(item => item.id === newsId);
      addNotification({
        id: Date.now().toString(),
        type: 'info',
        message: 'Bookmark functionality coming soon',
        timestamp: new Date().toISOString(),
        read: false,
      });
    } catch (error) {
      addNotification({
        id: Date.now().toString(),
        type: 'error',
        message: 'Failed to bookmark news item',
        timestamp: new Date().toISOString(),
        read: false,
      });
    }
  };

  const handleShare = (newsItem: any, e: React.MouseEvent) => {
    e.stopPropagation();
    if (navigator.share) {
      navigator.share({
        title: newsItem.title,
        text: newsItem.summary,
        url: newsItem.url,
      });
    } else {
      navigator.clipboard.writeText(newsItem.url);
      addNotification({
        id: Date.now().toString(),
        type: 'success',
        message: 'News URL copied to clipboard',
        timestamp: new Date().toISOString(),
        read: false,
      });
    }
  };

  if (isLoading && news.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-center h-64">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
            <NewspaperIcon className="h-5 w-5 mr-2" />
            News Analysis
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {filteredNews.length} articles • Real-time updates
          </p>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="space-y-3">
        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search news..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2">
          {/* Sentiment Filter */}
          <select
            value={sentimentFilter}
            onChange={(e) => setSentimentFilter(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="all">All Sentiment</option>
            <option value="positive">Positive</option>
            <option value="negative">Negative</option>
            <option value="neutral">Neutral</option>
          </select>

          {/* Source Filter */}
          <select
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
            className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="all">All Sources</option>
            {uniqueSources.map((source) => (
              <option key={source} value={source}>{source}</option>
            ))}
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="publishedAt">Latest</option>
            <option value="relevance">Relevance</option>
            <option value="sentiment">Sentiment</option>
          </select>
        </div>
      </div>

      {/* News List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredNews.map((item) => (
          <div
            key={item.id}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-gray-300 dark:hover:border-gray-600 transition-colors cursor-pointer"
            onClick={() => setSelectedNewsItem(item)}
          >
            <div className="flex items-start space-x-3">
              {item.imageUrl && (
                <img
                  src={item.imageUrl}
                  alt={item.title}
                  className="w-16 h-16 object-cover rounded-lg flex-shrink-0"
                />
              )}
              
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white text-sm line-clamp-2">
                    {item.title}
                  </h4>
                  <div className="flex items-center space-x-1 ml-2">
                    {getSentimentIcon(item.sentiment)}
                  </div>
                </div>
                
                <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-2">
                  {item.summary}
                </p>
                
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-3 text-gray-500 dark:text-gray-400">
                    <div className="flex items-center space-x-1">
                      <ClockIcon className="h-3 w-3" />
                      <span>{isClient ? formatRelativeTime(item.publishedAt) : ''}</span>
                    </div>
                    <span>{item.source}</span>
                    <span className={getSentimentColor(item.sentiment)}>
                      {item.sentiment}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={(e) => handleLike(item.id, e)}
                      className="flex items-center space-x-1 text-gray-500 hover:text-red-500 transition-colors"
                    >
                      <HeartIcon className="h-3 w-3" />
                      <span>0</span>
                    </button>
                    <button
                      onClick={(e) => handleBookmark(item.id, e)}
                      className="flex items-center space-x-1 text-gray-500 hover:text-blue-500 transition-colors"
                    >
                      <BookmarkIcon className="h-3 w-3" />
                    </button>
                    <button
                      onClick={(e) => handleShare(item, e)}
                      className="flex items-center space-x-1 text-gray-500 hover:text-green-500 transition-colors"
                    >
                      <ShareIcon className="h-3 w-3" />
                    </button>
                  </div>
                </div>
                
                {/* Tags */}
                {item.relevantAssets && item.relevantAssets.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {item.relevantAssets.slice(0, 3).map((asset: string) => (
                    <span
                      key={asset}
                      className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full"
                    >
                      {asset}
                    </span>
                  ))}
                  {item.relevantAssets.length > 3 && (
                    <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full">
                      +{item.relevantAssets.length - 3} more
                    </span>
                  )}
                </div>
              )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredNews.length === 0 && (
        <div className="text-center py-8">
          <NewspaperIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            {searchQuery || sentimentFilter !== 'all' || sourceFilter !== 'all'
              ? 'No news articles found matching your criteria.'
              : 'No news articles available.'}
          </p>
        </div>
      )}

      {/* News Detail Modal */}
      {selectedNewsItem && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white pr-4">
                  {selectedNewsItem.title}
                </h2>
                <button
                  onClick={() => setSelectedNewsItem(null)}
                  className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 flex-shrink-0"
                >
                  <XMarkIcon className="h-6 w-6" />
                </button>
              </div>
              
              {selectedNewsItem.imageUrl && (
                <img
                  src={selectedNewsItem.imageUrl}
                  alt={selectedNewsItem.title}
                  className="w-full h-48 object-cover rounded-lg mb-4"
                />
              )}
              
              <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400 mb-4">
                <span>{selectedNewsItem.source}</span>
                {selectedNewsItem.author && <span>by {selectedNewsItem.author}</span>}
                <span>{isClient ? formatRelativeTime(selectedNewsItem.publishedAt) : new Date(selectedNewsItem.publishedAt).toLocaleDateString()}</span>
                <div className="flex items-center space-x-1">
                  {getSentimentIcon(selectedNewsItem.sentiment)}
                  <span className={getSentimentColor(selectedNewsItem.sentiment)}>
                    {selectedNewsItem.sentiment}
                  </span>
                </div>
              </div>
              
              <p className="text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">
                {selectedNewsItem.content || selectedNewsItem.summary}
              </p>
              
              {selectedNewsItem.relevantAssets && selectedNewsItem.relevantAssets.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {selectedNewsItem.relevantAssets.map((asset: string) => (
                    <span
                      key={asset}
                      className="px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full"
                    >
                      {asset}
                    </span>
                  ))}
                </div>
              )}
              
              <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-4">
                  <button
                    onClick={(e) => handleLike(selectedNewsItem.id, e)}
                    className="flex items-center space-x-2 text-gray-500 hover:text-red-500 transition-colors"
                  >
                    <HeartIcon className="h-5 w-5" />
                    <span>0</span>
                  </button>
                  <div className="flex items-center space-x-2 text-gray-500">
                    <ChatBubbleLeftIcon className="h-5 w-5" />
                    <span>0</span>
                  </div>
                  <div className="flex items-center space-x-2 text-gray-500">
                    <EyeIcon className="h-5 w-5" />
                    <span>0</span>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => handleBookmark(selectedNewsItem.id, e)}
                    className={`p-2 rounded-lg transition-colors ${
                      selectedNewsItem.isBookmarked
                        ? 'bg-primary-100 text-primary-600'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {selectedNewsItem.isBookmarked ? (
                      <BookmarkIconSolid className="h-5 w-5" />
                    ) : (
                      <BookmarkIcon className="h-5 w-5" />
                    )}
                  </button>
                  <button 
                    onClick={(e) => handleShare(selectedNewsItem, e)}
                    className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors"
                  >
                    <ShareIcon className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NewsAnalysis;