'use client';

import React, { useEffect } from 'react';
import {
  SparklesIcon,
  LightBulbIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChartBarIcon,
  ClockIcon,
  ArrowPathIcon,
  BookmarkIcon,
  ShareIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';
import { BookmarkIcon as BookmarkIconSolid } from '@heroicons/react/24/solid';
import { useStore } from '../store/useStore';
import { formatRelativeTime, formatCurrency, formatPercentage } from '../utils';
import LoadingSpinner from './ui/LoadingSpinner';

interface AIInsight {
  id: string;
  type: 'prediction' | 'analysis' | 'recommendation' | 'alert' | 'trend';
  title: string;
  description: string;
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  timeframe: string;
  relatedAssets: string[];
  tags: string[];
  timestamp: string;
  isBookmarked?: boolean;
  views?: number;
  source: string;
  data?: {
    currentPrice?: number;
    targetPrice?: number;
    priceChange?: number;
    probability?: number;
    riskLevel?: string;
    [key: string]: any;
  };
}

const AIInsights: React.FC = () => {
  const {
    selectedAssets,
    selectedModels,
    isLoading,
    addNotification,
  } = useStore();
  
  const [insights, setInsights] = React.useState<AIInsight[]>([]);
  const [filteredInsights, setFilteredInsights] = React.useState<AIInsight[]>([]);
  const [typeFilter, setTypeFilter] = React.useState<'all' | AIInsight['type']>('all');
  const [impactFilter, setImpactFilter] = React.useState<'all' | AIInsight['impact']>('all');
  const [sortBy, setSortBy] = React.useState<'timestamp' | 'confidence' | 'impact'>('timestamp');
  const [selectedInsight, setSelectedInsight] = React.useState<AIInsight | null>(null);
  const [isRefreshing, setIsRefreshing] = React.useState(false);

  // Generate mock AI insights
  const generateMockInsights = React.useCallback((): AIInsight[] => {
    const insightTemplates = [
      {
        type: 'prediction' as const,
        title: 'Bitcoin Price Surge Expected',
        description: 'Technical analysis indicates a potential 15% price increase within the next 7 days based on current market momentum and institutional buying patterns.',
        confidence: 0.85,
        impact: 'high' as const,
        timeframe: '7 days',
        relatedAssets: ['BTC'],
        tags: ['Technical Analysis', 'Price Prediction', 'Bullish'],
        source: 'ML Price Predictor',
        data: {
          currentPrice: 45000,
          targetPrice: 51750,
          priceChange: 15,
          probability: 0.85,
        },
      },
      {
        type: 'analysis' as const,
        title: 'Ethereum Network Activity Analysis',
        description: 'On-chain metrics show increased DeFi activity and staking participation, indicating strong network fundamentals and potential price support.',
        confidence: 0.78,
        impact: 'medium' as const,
        timeframe: '30 days',
        relatedAssets: ['ETH'],
        tags: ['On-chain Analysis', 'DeFi', 'Network Activity'],
        source: 'Blockchain Analytics',
        data: {
          currentPrice: 2800,
          riskLevel: 'low',
        },
      },
      {
        type: 'recommendation' as const,
        title: 'Portfolio Rebalancing Suggestion',
        description: 'Current portfolio allocation shows overexposure to large-cap assets. Consider diversifying into mid-cap altcoins for better risk-adjusted returns.',
        confidence: 0.72,
        impact: 'medium' as const,
        timeframe: '14 days',
        relatedAssets: ['BTC', 'ETH', 'ADA', 'SOL'],
        tags: ['Portfolio Management', 'Risk Management', 'Diversification'],
        source: 'Portfolio Optimizer',
      },
      {
        type: 'alert' as const,
        title: 'High Volatility Warning',
        description: 'Market volatility has increased significantly. Consider reducing position sizes or implementing stop-loss orders to protect against sudden price movements.',
        confidence: 0.91,
        impact: 'high' as const,
        timeframe: 'Immediate',
        relatedAssets: ['BTC', 'ETH', 'ADA'],
        tags: ['Risk Alert', 'Volatility', 'Risk Management'],
        source: 'Risk Monitor',
        data: {
          riskLevel: 'high',
        },
      },
      {
        type: 'trend' as const,
        title: 'Emerging DeFi Trend Detected',
        description: 'Social sentiment analysis reveals growing interest in yield farming protocols. Early adoption could present profitable opportunities.',
        confidence: 0.68,
        impact: 'medium' as const,
        timeframe: '21 days',
        relatedAssets: ['ETH', 'LINK'],
        tags: ['Trend Analysis', 'DeFi', 'Social Sentiment'],
        source: 'Trend Detector',
      },
    ];

    return insightTemplates.map((template, index) => ({
      id: `insight-${index + 1}`,
      ...template,
      timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
      isBookmarked: Math.random() > 0.7,
      views: Math.floor(Math.random() * 500) + 50,
    }));
  }, []);

  // Initialize insights
  useEffect(() => {
    const mockInsights = generateMockInsights();
    setInsights(mockInsights);
  }, [generateMockInsights]);

  // Filter insights based on selected assets and models
  useEffect(() => {
    let filtered = insights.filter(insight => {
      const matchesType = typeFilter === 'all' || insight.type === typeFilter;
      const matchesImpact = impactFilter === 'all' || insight.impact === impactFilter;
      
      const matchesAssets = selectedAssets.length === 0 || 
                           selectedAssets.some(asset => 
                             insight.relatedAssets.includes(asset.symbol) || 
                             insight.relatedAssets.includes(asset.id)
                           );
      
      return matchesType && matchesImpact && matchesAssets;
    });

    // Sort insights
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'timestamp':
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        case 'confidence':
          return b.confidence - a.confidence;
        case 'impact':
          const impactOrder = { high: 3, medium: 2, low: 1 };
          return impactOrder[b.impact] - impactOrder[a.impact];
        default:
          return 0;
      }
    });

    setFilteredInsights(filtered);
  }, [insights, typeFilter, impactFilter, sortBy, selectedAssets]);

  const getInsightIcon = (type: AIInsight['type']) => {
    switch (type) {
      case 'prediction':
        return <ArrowUpIcon className="h-5 w-5 text-blue-500" />;
      case 'analysis':
        return <ChartBarIcon className="h-5 w-5 text-green-500" />;
      case 'recommendation':
        return <LightBulbIcon className="h-5 w-5 text-yellow-500" />;
      case 'alert':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      case 'trend':
        return <SparklesIcon className="h-5 w-5 text-purple-500" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getImpactColor = (impact: AIInsight['impact']) => {
    switch (impact) {
      case 'high':
        return 'text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/20';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/20';
      case 'low':
        return 'text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/20';
      default:
        return 'text-gray-600 bg-gray-100 dark:text-gray-400 dark:bg-gray-900/20';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 dark:text-green-400';
    if (confidence >= 0.6) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      const newInsights = generateMockInsights();
      setInsights(newInsights);
      addNotification({
        id: Date.now().toString(),
        type: 'success',
        message: 'AI insights refreshed successfully',
        read: false,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      addNotification({
        id: Date.now().toString(),
        type: 'error',
        message: 'Failed to refresh AI insights',
        read: false,
        timestamp: new Date().toISOString(),
      });
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleBookmark = (insightId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setInsights(prev => prev.map(insight => 
      insight.id === insightId 
        ? { ...insight, isBookmarked: !insight.isBookmarked }
        : insight
    ));
    
    const insight = insights.find(i => i.id === insightId);
    addNotification({
      id: Date.now().toString(),
      type: 'success',
      message: `Insight ${insight?.isBookmarked ? 'removed from' : 'added to'} bookmarks`,
      timestamp: new Date().toISOString(),
      read: false,
    });
  };

  const handleShare = (insight: AIInsight, e: React.MouseEvent) => {
    e.stopPropagation();
    const shareText = `${insight.title}: ${insight.description}`;
    
    if (navigator.share) {
      navigator.share({
        title: insight.title,
        text: shareText,
      });
    } else {
      navigator.clipboard.writeText(shareText);
      addNotification({
        id: Date.now().toString(),
        type: 'success',
        message: 'Insight copied to clipboard',
        timestamp: new Date().toISOString(),
        read: false,
      });
    }
  };

  if (isLoading && insights.length === 0) {
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
            <SparklesIcon className="h-5 w-5 mr-2" />
            AI Insights
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {filteredInsights.length} insights • Powered by ML models
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center space-x-2 px-3 py-2 text-sm bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <ArrowPathIcon className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value as any)}
          className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="all">All Types</option>
          <option value="prediction">Predictions</option>
          <option value="analysis">Analysis</option>
          <option value="recommendation">Recommendations</option>
          <option value="alert">Alerts</option>
          <option value="trend">Trends</option>
        </select>

        <select
          value={impactFilter}
          onChange={(e) => setImpactFilter(e.target.value as any)}
          className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="all">All Impact</option>
          <option value="high">High Impact</option>
          <option value="medium">Medium Impact</option>
          <option value="low">Low Impact</option>
        </select>

        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
          className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="timestamp">Latest</option>
          <option value="confidence">Confidence</option>
          <option value="impact">Impact</option>
        </select>
      </div>

      {/* Insights List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredInsights.map((insight) => (
          <div
            key={insight.id}
            className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-gray-300 dark:hover:border-gray-600 transition-colors cursor-pointer"
            onClick={() => setSelectedInsight(insight)}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-start space-x-3">
                {getInsightIcon(insight.type)}
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900 dark:text-white text-sm">
                    {insight.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
                    {insight.description}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 text-xs rounded-full ${getImpactColor(insight.impact)}`}>
                  {insight.impact}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-4 text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-1">
                  <ClockIcon className="h-3 w-3" />
                  <span>{formatRelativeTime(insight.timestamp)}</span>
                </div>
                <span>Confidence: <span className={getConfidenceColor(insight.confidence)}>{Math.round(insight.confidence * 100)}%</span></span>
                <span>{insight.timeframe}</span>
                <span>{insight.source}</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-1 text-gray-500">
                  <EyeIcon className="h-3 w-3" />
                  <span>{insight.views || 0}</span>
                </div>
                <button
                  onClick={(e) => handleBookmark(insight.id, e)}
                  className={`transition-colors ${
                    insight.isBookmarked
                      ? 'text-primary-500 hover:text-primary-600'
                      : 'text-gray-500 hover:text-primary-500'
                  }`}
                >
                  {insight.isBookmarked ? (
                    <BookmarkIconSolid className="h-3 w-3" />
                  ) : (
                    <BookmarkIcon className="h-3 w-3" />
                  )}
                </button>
                <button
                  onClick={(e) => handleShare(insight, e)}
                  className="text-gray-500 hover:text-primary-500 transition-colors"
                >
                  <ShareIcon className="h-3 w-3" />
                </button>
              </div>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-1 mt-2">
              {insight.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full"
                >
                  {tag}
                </span>
              ))}
              {insight.tags.length > 3 && (
                <span className="px-2 py-1 text-xs text-gray-500 dark:text-gray-400">
                  +{insight.tags.length - 3} more
                </span>
              )}
            </div>

            {/* Data Preview */}
            {insight.data && (
              <div className="mt-3 p-2 bg-gray-50 dark:bg-gray-700/50 rounded text-xs">
                <div className="flex flex-wrap gap-3">
                  {insight.data.currentPrice && (
                    <span>Current: {formatCurrency(insight.data.currentPrice)}</span>
                  )}
                  {insight.data.targetPrice && (
                    <span>Target: {formatCurrency(insight.data.targetPrice)}</span>
                  )}
                  {insight.data.priceChange && (
                    <span className={insight.data.priceChange > 0 ? 'text-green-600' : 'text-red-600'}>
                      Change: {formatPercentage(insight.data.priceChange / 100)}
                    </span>
                  )}
                  {insight.data.probability && (
                    <span>Probability: {Math.round(insight.data.probability * 100)}%</span>
                  )}
                  {insight.data.riskLevel && (
                    <span className={`capitalize ${
                      insight.data.riskLevel === 'high' ? 'text-red-600' :
                      insight.data.riskLevel === 'medium' ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      Risk: {insight.data.riskLevel}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredInsights.length === 0 && (
        <div className="text-center py-8">
          <SparklesIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            {typeFilter !== 'all' || impactFilter !== 'all'
              ? 'No insights found matching your criteria.'
              : 'No AI insights available. Try selecting some assets or models.'}
          </p>
        </div>
      )}

      {/* Insight Detail Modal */}
      {selectedInsight && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-3">
                  {getInsightIcon(selectedInsight.type)}
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                      {selectedInsight.title}
                    </h2>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={`px-2 py-1 text-xs rounded-full ${getImpactColor(selectedInsight.impact)}`}>
                        {selectedInsight.impact} impact
                      </span>
                      <span className="text-sm text-gray-500">
                        {selectedInsight.type}
                      </span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedInsight(null)}
                  className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                >
                  ×
                </button>
              </div>
              
              <p className="text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">
                {selectedInsight.description}
              </p>
              
              <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                <div>
                  <span className="text-gray-500">Confidence:</span>
                  <span className={`ml-2 font-semibold ${getConfidenceColor(selectedInsight.confidence)}`}>
                    {Math.round(selectedInsight.confidence * 100)}%
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Timeframe:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {selectedInsight.timeframe}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Source:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {selectedInsight.source}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Generated:</span>
                  <span className="ml-2 font-semibold text-gray-900 dark:text-white">
                    {formatRelativeTime(selectedInsight.timestamp)}
                  </span>
                </div>
              </div>

              {selectedInsight.data && Object.keys(selectedInsight.data).length > 0 && (
                <div className="mb-4">
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Data Points</h3>
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      {selectedInsight.data.currentPrice && (
                        <div>
                          <span className="text-gray-500">Current Price:</span>
                          <span className="ml-2 font-semibold">{formatCurrency(selectedInsight.data.currentPrice)}</span>
                        </div>
                      )}
                      {selectedInsight.data.targetPrice && (
                        <div>
                          <span className="text-gray-500">Target Price:</span>
                          <span className="ml-2 font-semibold">{formatCurrency(selectedInsight.data.targetPrice)}</span>
                        </div>
                      )}
                      {selectedInsight.data.priceChange && (
                        <div>
                          <span className="text-gray-500">Expected Change:</span>
                          <span className={`ml-2 font-semibold ${
                            selectedInsight.data.priceChange > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {formatPercentage(selectedInsight.data.priceChange / 100)}
                          </span>
                        </div>
                      )}
                      {selectedInsight.data.probability && (
                        <div>
                          <span className="text-gray-500">Probability:</span>
                          <span className="ml-2 font-semibold">{Math.round(selectedInsight.data.probability * 100)}%</span>
                        </div>
                      )}
                      {selectedInsight.data.riskLevel && (
                        <div>
                          <span className="text-gray-500">Risk Level:</span>
                          <span className={`ml-2 font-semibold capitalize ${
                            selectedInsight.data.riskLevel === 'high' ? 'text-red-600' :
                            selectedInsight.data.riskLevel === 'medium' ? 'text-yellow-600' :
                            'text-green-600'
                          }`}>
                            {selectedInsight.data.riskLevel}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              <div className="flex flex-wrap gap-2 mb-4">
                {selectedInsight.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              
              <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-4 text-sm text-gray-500">
                  <div className="flex items-center space-x-1">
                    <EyeIcon className="h-4 w-4" />
                    <span>{selectedInsight.views || 0} views</span>
                  </div>
                  <span>Related: {selectedInsight.relatedAssets.join(', ')}</span>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button
                    onClick={(e) => handleBookmark(selectedInsight.id, e)}
                    className={`p-2 rounded-lg transition-colors ${
                      selectedInsight.isBookmarked
                        ? 'bg-primary-100 text-primary-600'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {selectedInsight.isBookmarked ? (
                      <BookmarkIconSolid className="h-5 w-5" />
                    ) : (
                      <BookmarkIcon className="h-5 w-5" />
                    )}
                  </button>
                  <button 
                    onClick={(e) => handleShare(selectedInsight, e)}
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

export default AIInsights;