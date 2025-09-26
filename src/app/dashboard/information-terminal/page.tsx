'use client';

import React, { useState, useEffect } from 'react';
import NewsPanel from '@/components/NewsPanel';
import NewsAnalysis from '@/components/NewsAnalysis';

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  publishedAt: string;
  url: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  symbols: string[];
}

interface SocialPost {
  id: string;
  platform: 'reddit' | 'twitter' | 'discord' | 'tiktok';
  content: string;
  author: string;
  publishedAt: string;
  engagement: number;
  sentiment: 'positive' | 'negative' | 'neutral';
  symbols: string[];
}

export default function InformationTerminalPage() {
  const [selectedTab, setSelectedTab] = useState<'news' | 'social'>('news');
  const [selectedSource, setSelectedSource] = useState<string>('all');
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [socialPosts, setSocialPosts] = useState<SocialPost[]>([]);
  const [selectedItem, setSelectedItem] = useState<NewsItem | SocialPost | null>(null);
  const [watchlistSymbols, setWatchlistSymbols] = useState<string[]>(['AAPL', 'TSLA', 'NVDA', 'BTC', 'ETH']);

  const newsSources = [
    { value: 'all', label: 'All Sources' },
    { value: 'reuters', label: 'Reuters' },
    { value: 'bloomberg', label: 'Bloomberg' },
    { value: 'cnbc', label: 'CNBC' },
    { value: 'marketwatch', label: 'MarketWatch' },
    { value: 'yahoo', label: 'Yahoo Finance' }
  ];

  const socialPlatforms = [
    { value: 'all', label: 'All Platforms' },
    { value: 'reddit', label: 'Reddit' },
    { value: 'twitter', label: 'Twitter/X' },
    { value: 'discord', label: 'Discord' },
    { value: 'tiktok', label: 'TikTok' }
  ];

  useEffect(() => {
    // Mock data - replace with actual API calls
    setNewsItems([
      {
        id: '1',
        title: 'Apple Reports Strong Q4 Earnings',
        summary: 'Apple Inc. reported better-than-expected quarterly earnings...',
        source: 'Reuters',
        publishedAt: '2024-01-15T10:30:00Z',
        url: 'https://example.com/news/1',
        sentiment: 'positive',
        symbols: ['AAPL']
      },
      {
        id: '2',
        title: 'Tesla Announces New Gigafactory',
        summary: 'Tesla announced plans for a new Gigafactory in Europe...',
        source: 'Bloomberg',
        publishedAt: '2024-01-15T09:15:00Z',
        url: 'https://example.com/news/2',
        sentiment: 'positive',
        symbols: ['TSLA']
      }
    ]);

    setSocialPosts([
      {
        id: '1',
        platform: 'reddit',
        content: 'NVDA looking strong after the latest AI chip announcement. Bullish sentiment across the board.',
        author: 'TechTrader2024',
        publishedAt: '2024-01-15T11:00:00Z',
        engagement: 245,
        sentiment: 'positive',
        symbols: ['NVDA']
      },
      {
        id: '2',
        platform: 'twitter',
        content: 'Bitcoin breaking through resistance levels. Could see $50k soon! 🚀',
        author: 'CryptoAnalyst',
        publishedAt: '2024-01-15T10:45:00Z',
        engagement: 1200,
        sentiment: 'positive',
        symbols: ['BTC']
      }
    ]);
  }, []);

  const filteredNews = newsItems.filter(item => 
    item.symbols.some(symbol => watchlistSymbols.includes(symbol)) &&
    (selectedSource === 'all' || item.source.toLowerCase().includes(selectedSource))
  );

  const filteredSocial = socialPosts.filter(post => 
    post.symbols.some(symbol => watchlistSymbols.includes(symbol)) &&
    (selectedSource === 'all' || post.platform === selectedSource)
  );

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left Panel - Feed */}
      <div className="w-1/2 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Information Terminal</h2>
          <p className="text-sm text-gray-400">Global News & Social Media Intelligence</p>
          
          {/* Tab Selector */}
          <div className="flex mt-4 space-x-2">
            <button
              onClick={() => setSelectedTab('news')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedTab === 'news'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              News Feed
            </button>
            <button
              onClick={() => setSelectedTab('social')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedTab === 'social'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Social Media
            </button>
          </div>

          {/* Source Filter */}
          <div className="mt-4">
            <select
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {selectedTab === 'news' 
                ? newsSources.map(source => (
                    <option key={source.value} value={source.value}>{source.label}</option>
                  ))
                : socialPlatforms.map(platform => (
                    <option key={platform.value} value={platform.value}>{platform.label}</option>
                  ))
              }
            </select>
          </div>
        </div>

        {/* Feed Content */}
        <div className="flex-1 overflow-y-auto">
          {selectedTab === 'news' ? (
            <div className="p-4 space-y-4">
              {filteredNews.map((item) => (
                <div
                  key={item.id}
                  onClick={() => setSelectedItem(item)}
                  className="bg-gray-700 rounded-lg p-4 cursor-pointer hover:bg-gray-600 transition-colors border-l-4 border-blue-500"
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold text-white text-sm">{item.title}</h3>
                    <span className={`px-2 py-1 rounded text-xs ${
                      item.sentiment === 'positive' ? 'bg-green-600' :
                      item.sentiment === 'negative' ? 'bg-red-600' : 'bg-gray-600'
                    }`}>
                      {item.sentiment}
                    </span>
                  </div>
                  <p className="text-gray-300 text-sm mb-2">{item.summary}</p>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>{item.source}</span>
                    <span>{new Date(item.publishedAt).toLocaleTimeString()}</span>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {item.symbols.map(symbol => (
                      <span key={symbol} className="bg-blue-600 text-white px-2 py-1 rounded text-xs">
                        {symbol}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 space-y-4">
              {filteredSocial.map((post) => (
                <div
                  key={post.id}
                  onClick={() => setSelectedItem(post)}
                  className="bg-gray-700 rounded-lg p-4 cursor-pointer hover:bg-gray-600 transition-colors border-l-4 border-purple-500"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-white text-sm">{post.author}</span>
                      <span className="bg-purple-600 text-white px-2 py-1 rounded text-xs">
                        {post.platform}
                      </span>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs ${
                      post.sentiment === 'positive' ? 'bg-green-600' :
                      post.sentiment === 'negative' ? 'bg-red-600' : 'bg-gray-600'
                    }`}>
                      {post.sentiment}
                    </span>
                  </div>
                  <p className="text-gray-300 text-sm mb-2">{post.content}</p>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>❤️ {post.engagement}</span>
                    <span>{new Date(post.publishedAt).toLocaleTimeString()}</span>
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {post.symbols.map(symbol => (
                      <span key={symbol} className="bg-blue-600 text-white px-2 py-1 rounded text-xs">
                        {symbol}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right Panel - Full Content */}
      <div className="w-1/2 bg-gray-900 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">
            {selectedItem ? 'Full Content' : 'Select an item to read'}
          </h3>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          {selectedItem ? (
            <div className="space-y-4">
              {'title' in selectedItem ? (
                // News Item
                <div>
                  <h2 className="text-xl font-bold text-white mb-4">{selectedItem.title}</h2>
                  <div className="flex items-center space-x-4 mb-4 text-sm text-gray-400">
                    <span>{selectedItem.source}</span>
                    <span>{new Date(selectedItem.publishedAt).toLocaleString()}</span>
                    <span className={`px-2 py-1 rounded ${
                      selectedItem.sentiment === 'positive' ? 'bg-green-600' :
                      selectedItem.sentiment === 'negative' ? 'bg-red-600' : 'bg-gray-600'
                    }`}>
                      {selectedItem.sentiment}
                    </span>
                  </div>
                  <div className="prose prose-invert max-w-none">
                    <p className="text-gray-300 leading-relaxed">
                      {selectedItem.summary}
                    </p>
                    <p className="text-gray-300 leading-relaxed mt-4">
                      This is where the full article content would be displayed. 
                      The content would be fetched from the original source and displayed 
                      within the app for seamless reading experience.
                    </p>
                  </div>
                  <div className="mt-6">
                    <a
                      href={selectedItem.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Read Original Article
                      <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  </div>
                </div>
              ) : (
                // Social Post
                <div>
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center">
                      <span className="text-white font-semibold">
                        {selectedItem.author.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">{selectedItem.author}</h3>
                      <div className="flex items-center space-x-2 text-sm text-gray-400">
                        <span className="bg-purple-600 text-white px-2 py-1 rounded text-xs">
                          {selectedItem.platform}
                        </span>
                        <span>{new Date(selectedItem.publishedAt).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-800 rounded-lg p-4 mb-4">
                    <p className="text-gray-300 leading-relaxed">{selectedItem.content}</p>
                  </div>
                  
                  <div className="flex items-center justify-between text-sm text-gray-400">
                    <div className="flex items-center space-x-4">
                      <span>❤️ {selectedItem.engagement} engagements</span>
                      <span className={`px-2 py-1 rounded ${
                        selectedItem.sentiment === 'positive' ? 'bg-green-600' :
                        selectedItem.sentiment === 'negative' ? 'bg-red-600' : 'bg-gray-600'
                      }`}>
                        {selectedItem.sentiment}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex flex-wrap gap-2 mt-4">
                    {selectedItem.symbols.map(symbol => (
                      <span key={symbol} className="bg-blue-600 text-white px-3 py-1 rounded">
                        {symbol}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                </svg>
                <p>Select a news article or social media post to read the full content</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}