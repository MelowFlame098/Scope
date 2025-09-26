'use client'

import { useState, useEffect } from 'react'
import {
  NewspaperIcon,
  ClockIcon,
  FireIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  GlobeAltIcon,
  FunnelIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline'

interface NewsArticle {
  id: string
  title: string
  summary: string
  content: string
  source: string
  author?: string
  publishedAt: string
  url: string
  imageUrl?: string
  category: 'crypto' | 'stocks' | 'forex' | 'commodities' | 'general'
  sentiment: 'positive' | 'negative' | 'neutral'
  sentimentScore: number
  relevantAssets: string[]
  tags: string[]
  readTime: number
}

const mockNews: NewsArticle[] = [
  {
    id: '1',
    title: 'Bitcoin Surges Past $45,000 as Institutional Adoption Accelerates',
    summary: 'Major financial institutions continue to embrace Bitcoin, driving the cryptocurrency to new monthly highs amid growing institutional demand.',
    content: 'Bitcoin has broken through the $45,000 resistance level following announcements from several major financial institutions regarding their cryptocurrency adoption strategies...',
    source: 'CryptoNews Today',
    author: 'Sarah Johnson',
    publishedAt: '2024-01-15T10:30:00Z',
    url: 'https://example.com/bitcoin-surge',
    imageUrl: '/api/placeholder/400/200',
    category: 'crypto',
    sentiment: 'positive',
    sentimentScore: 0.85,
    relevantAssets: ['BTC', 'ETH'],
    tags: ['institutional', 'adoption', 'bullish'],
    readTime: 3
  },
  {
    id: '2',
    title: 'Federal Reserve Signals Potential Rate Cuts in Q2 2024',
    summary: 'Fed officials hint at possible monetary policy shifts as inflation shows signs of cooling, potentially impacting equity markets.',
    content: 'Federal Reserve officials have indicated that interest rate cuts may be on the table for the second quarter of 2024...',
    source: 'Financial Times',
    author: 'Michael Chen',
    publishedAt: '2024-01-15T09:15:00Z',
    url: 'https://example.com/fed-rate-cuts',
    category: 'general',
    sentiment: 'positive',
    sentimentScore: 0.72,
    relevantAssets: ['SPY', 'QQQ', 'USD'],
    tags: ['federal-reserve', 'interest-rates', 'monetary-policy'],
    readTime: 4
  },
  {
    id: '3',
    title: 'Tesla Stock Drops 5% Following Production Concerns',
    summary: 'Tesla shares decline after reports of potential production delays at the Austin Gigafactory raise investor concerns.',
    content: 'Tesla Inc. shares fell sharply in pre-market trading following reports of potential production bottlenecks...',
    source: 'MarketWatch',
    author: 'Lisa Rodriguez',
    publishedAt: '2024-01-15T08:45:00Z',
    url: 'https://example.com/tesla-production',
    category: 'stocks',
    sentiment: 'negative',
    sentimentScore: -0.68,
    relevantAssets: ['TSLA'],
    tags: ['tesla', 'production', 'earnings'],
    readTime: 2
  },
  {
    id: '4',
    title: 'EUR/USD Reaches 3-Month High on ECB Policy Expectations',
    summary: 'The Euro strengthens against the Dollar as markets anticipate European Central Bank policy announcements.',
    content: 'The EUR/USD currency pair has climbed to its highest level in three months as traders position ahead of the ECB meeting...',
    source: 'Forex Daily',
    author: 'James Wilson',
    publishedAt: '2024-01-15T07:20:00Z',
    url: 'https://example.com/eurusd-high',
    category: 'forex',
    sentiment: 'neutral',
    sentimentScore: 0.15,
    relevantAssets: ['EURUSD', 'EUR', 'USD'],
    tags: ['ecb', 'currency', 'policy'],
    readTime: 3
  },
  {
    id: '5',
    title: 'Gold Prices Stabilize Amid Geopolitical Tensions',
    summary: 'Precious metals maintain steady levels as investors seek safe-haven assets during uncertain times.',
    content: 'Gold prices have found support around the $2,040 level as geopolitical tensions continue to influence market sentiment...',
    source: 'Commodities Weekly',
    author: 'David Kim',
    publishedAt: '2024-01-15T06:30:00Z',
    url: 'https://example.com/gold-stabilize',
    category: 'commodities',
    sentiment: 'neutral',
    sentimentScore: 0.05,
    relevantAssets: ['XAU', 'GLD'],
    tags: ['gold', 'safe-haven', 'geopolitical'],
    readTime: 2
  }
]

const categories = {
  all: { name: 'All News', icon: NewspaperIcon },
  crypto: { name: 'Crypto', icon: FireIcon },
  stocks: { name: 'Stocks', icon: ArrowUpIcon },
  forex: { name: 'Forex', icon: GlobeAltIcon },
  commodities: { name: 'Commodities', icon: ArrowDownIcon },
  general: { name: 'General', icon: NewspaperIcon }
}

const sentimentColors = {
  positive: 'text-green-400',
  negative: 'text-red-400',
  neutral: 'text-yellow-400'
}

export default function NewsPanel() {
  const [news] = useState<NewsArticle[]>(mockNews)
  const [filteredNews, setFilteredNews] = useState<NewsArticle[]>(mockNews)
  const [activeCategory, setActiveCategory] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [sentimentFilter, setSentimentFilter] = useState<string>('all')
  const [isClient, setIsClient] = useState(false)
  const [expandedArticle, setExpandedArticle] = useState<string | null>(null)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    let filtered = news

    // Category filter
    if (activeCategory !== 'all') {
      filtered = filtered.filter(article => article.category === activeCategory)
    }

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(article => 
        article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        article.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
        article.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    }

    // Sentiment filter
    if (sentimentFilter !== 'all') {
      filtered = filtered.filter(article => article.sentiment === sentimentFilter)
    }

    setFilteredNews(filtered)
  }, [news, activeCategory, searchQuery, sentimentFilter])

  const formatTimestamp = (timestamp: string): string => {
    if (!isClient) {
      return ''
    }
    
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    return date.toLocaleDateString()
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return <ArrowUpIcon className="h-4 w-4 text-green-400" />
      case 'negative': return <ArrowDownIcon className="h-4 w-4 text-red-400" />
      default: return <div className="h-4 w-4 rounded-full bg-yellow-400" />
    }
  }

  const getSentimentScore = (score: number): string => {
    return `${score > 0 ? '+' : ''}${(score * 100).toFixed(0)}%`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center">
            <NewspaperIcon className="h-6 w-6 mr-2" />
            Financial News
          </h2>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-dark-400">{filteredNews.length} articles</span>
          </div>
        </div>

        {/* Filters */}
        <div className="space-y-4">
          {/* Search */}
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-dark-400" />
            <input
              type="text"
              placeholder="Search news..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-field pl-10"
            />
          </div>

          {/* Category and Sentiment Filters */}
          <div className="flex flex-wrap gap-2">
            {/* Category Filter */}
            <div className="flex space-x-1">
              {Object.entries(categories).map(([key, cat]) => {
                const IconComponent = cat.icon
                return (
                  <button
                    key={key}
                    onClick={() => setActiveCategory(key)}
                    className={`flex items-center space-x-1 px-3 py-1 text-sm rounded transition-colors ${
                      activeCategory === key
                        ? 'bg-primary-600 text-white'
                        : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                    }`}
                  >
                    <IconComponent className="h-4 w-4" />
                    <span>{cat.name}</span>
                  </button>
                )
              })}
            </div>

            {/* Sentiment Filter */}
            <select
              value={sentimentFilter}
              onChange={(e) => setSentimentFilter(e.target.value)}
              className="bg-dark-700 border border-dark-600 text-white text-sm rounded px-2 py-1"
            >
              <option value="all">All Sentiment</option>
              <option value="positive">Positive</option>
              <option value="neutral">Neutral</option>
              <option value="negative">Negative</option>
            </select>
          </div>
        </div>
      </div>

      {/* News Articles */}
      <div className="space-y-4">
        {filteredNews.length === 0 ? (
          <div className="card text-center py-8">
            <NewspaperIcon className="h-12 w-12 text-dark-500 mx-auto mb-2" />
            <p className="text-dark-400">No articles found</p>
            <p className="text-dark-500 text-sm">Try adjusting your filters</p>
          </div>
        ) : (
          filteredNews.map(article => {
            const isExpanded = expandedArticle === article.id
            
            return (
              <div
                key={article.id}
                className={`card cursor-pointer transition-all duration-200 ${
                  isExpanded ? 'ring-2 ring-primary-500' : 'hover:bg-dark-750'
                }`}
                onClick={() => setExpandedArticle(isExpanded ? null : article.id)}
              >
                <div className="flex items-start space-x-4">
                  {/* Article Image */}
                  {article.imageUrl && (
                    <div className="w-24 h-16 bg-dark-600 rounded flex-shrink-0">
                      <div className="w-full h-full bg-gradient-to-br from-primary-600 to-primary-800 rounded flex items-center justify-center">
                        <NewspaperIcon className="h-6 w-6 text-white" />
                      </div>
                    </div>
                  )}

                  {/* Article Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="text-white font-semibold mb-1 line-clamp-2">
                          {article.title}
                        </h3>
                        
                        <p className="text-dark-300 text-sm mb-2 line-clamp-2">
                          {article.summary}
                        </p>
                        
                        <div className="flex items-center space-x-4 text-xs text-dark-400">
                          <div className="flex items-center space-x-1">
                            <ClockIcon className="h-3 w-3" />
                            <span>{formatTimestamp(article.publishedAt)}</span>
                          </div>
                          
                          <span>{article.source}</span>
                          
                          {article.author && (
                            <span>by {article.author}</span>
                          )}
                          
                          <span>{article.readTime} min read</span>
                        </div>
                      </div>
                      
                      {/* Sentiment Indicator */}
                      <div className="flex items-center space-x-2 ml-4">
                        {getSentimentIcon(article.sentiment)}
                        <span className={`text-xs ${sentimentColors[article.sentiment]}`}>
                          {getSentimentScore(article.sentimentScore)}
                        </span>
                      </div>
                    </div>

                    {/* Tags and Assets */}
                    <div className="flex items-center justify-between mt-3">
                      <div className="flex flex-wrap gap-1">
                        {article.relevantAssets.slice(0, 3).map(asset => (
                          <span
                            key={asset}
                            className="px-2 py-1 bg-primary-900/30 text-primary-400 text-xs rounded"
                          >
                            {asset}
                          </span>
                        ))}
                      </div>
                      
                      <div className="flex flex-wrap gap-1">
                        {article.tags.slice(0, 2).map(tag => (
                          <span
                            key={tag}
                            className="px-2 py-1 bg-dark-600 text-dark-300 text-xs rounded"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Expanded Content */}
                    {isExpanded && (
                      <div className="mt-4 pt-4 border-t border-dark-600">
                        <p className="text-dark-200 text-sm leading-relaxed mb-4">
                          {article.content}
                        </p>
                        
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <span className="text-sm text-dark-400">Sentiment Analysis:</span>
                            <div className="flex items-center space-x-2">
                              {getSentimentIcon(article.sentiment)}
                              <span className={`text-sm font-medium ${sentimentColors[article.sentiment]}`}>
                                {article.sentiment.charAt(0).toUpperCase() + article.sentiment.slice(1)}
                              </span>
                              <span className="text-dark-400">({getSentimentScore(article.sentimentScore)})</span>
                            </div>
                          </div>
                          
                          <a
                            href={article.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-primary text-sm"
                            onClick={(e) => e.stopPropagation()}
                          >
                            Read Full Article
                          </a>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* News Summary */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">News Summary</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-dark-700 rounded">
            <div className="text-2xl font-bold text-green-400">
              {filteredNews.filter(a => a.sentiment === 'positive').length}
            </div>
            <p className="text-white font-medium">Positive</p>
            <p className="text-dark-400 text-sm">Bullish News</p>
          </div>
          
          <div className="text-center p-4 bg-dark-700 rounded">
            <div className="text-2xl font-bold text-yellow-400">
              {filteredNews.filter(a => a.sentiment === 'neutral').length}
            </div>
            <p className="text-white font-medium">Neutral</p>
            <p className="text-dark-400 text-sm">Balanced Coverage</p>
          </div>
          
          <div className="text-center p-4 bg-dark-700 rounded">
            <div className="text-2xl font-bold text-red-400">
              {filteredNews.filter(a => a.sentiment === 'negative').length}
            </div>
            <p className="text-white font-medium">Negative</p>
            <p className="text-dark-400 text-sm">Bearish News</p>
          </div>
        </div>
      </div>
    </div>
  )
}