'use client';

import React, { useState, useEffect } from 'react';

interface TradeEntry {
  id: string;
  symbol: string;
  type: 'long' | 'short';
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  entryDate: string;
  exitDate?: string;
  pnl?: number;
  status: 'open' | 'closed';
  notes: string;
  tags: string[];
}

interface JournalEntry {
  id: string;
  title: string;
  content: string;
  date: string;
  tags: string[];
  mood: 'bullish' | 'bearish' | 'neutral';
  marketConditions: string;
  lessons: string;
}

interface Report {
  id: string;
  title: string;
  content: string;
  createdDate: string;
  lastModified: string;
  type: 'analysis' | 'strategy' | 'review';
  tags: string[];
}

export default function ArchivePage() {
  const [activeTab, setActiveTab] = useState<'trades' | 'journal' | 'reports'>('trades');
  const [trades, setTrades] = useState<TradeEntry[]>([]);
  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedItem, setSelectedItem] = useState<TradeEntry | JournalEntry | Report | null>(null);
  const [isCreating, setIsCreating] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterTag, setFilterTag] = useState<string>('all');

  useEffect(() => {
    // Mock data - replace with actual API calls
    setTrades([
      {
        id: '1',
        symbol: 'AAPL',
        type: 'long',
        entryPrice: 180.50,
        exitPrice: 185.25,
        quantity: 100,
        entryDate: '2024-01-10T09:30:00Z',
        exitDate: '2024-01-12T15:45:00Z',
        pnl: 475.00,
        status: 'closed',
        notes: 'Strong earnings beat, good technical setup',
        tags: ['earnings', 'tech', 'swing']
      },
      {
        id: '2',
        symbol: 'TSLA',
        type: 'long',
        entryPrice: 240.00,
        quantity: 50,
        entryDate: '2024-01-15T10:00:00Z',
        status: 'open',
        notes: 'Breakout above resistance, watching for continuation',
        tags: ['breakout', 'momentum', 'ev']
      }
    ]);

    setJournalEntries([
      {
        id: '1',
        title: 'Market Analysis - January 15, 2024',
        content: 'Today the market showed strong bullish momentum...',
        date: '2024-01-15T18:00:00Z',
        tags: ['market-analysis', 'bullish'],
        mood: 'bullish',
        marketConditions: 'Strong uptrend, low volatility',
        lessons: 'Patience paid off waiting for the right setup'
      },
      {
        id: '2',
        title: 'Trading Psychology Notes',
        content: 'Need to work on position sizing and risk management...',
        date: '2024-01-14T20:30:00Z',
        tags: ['psychology', 'risk-management'],
        mood: 'neutral',
        marketConditions: 'Choppy, sideways action',
        lessons: 'Emotional trading led to poor decisions'
      }
    ]);

    setReports([
      {
        id: '1',
        title: 'Q4 2023 Trading Performance Review',
        content: 'Comprehensive analysis of Q4 trading performance...',
        createdDate: '2024-01-01T12:00:00Z',
        lastModified: '2024-01-02T14:30:00Z',
        type: 'review',
        tags: ['performance', 'quarterly', 'review']
      },
      {
        id: '2',
        title: 'Momentum Trading Strategy',
        content: 'Detailed strategy for momentum-based trades...',
        createdDate: '2024-01-10T16:00:00Z',
        lastModified: '2024-01-10T16:00:00Z',
        type: 'strategy',
        tags: ['momentum', 'strategy', 'technical']
      }
    ]);
  }, []);

  const getAllTags = () => {
    const allTags = new Set<string>();
    if (activeTab === 'trades') {
      trades.forEach(trade => trade.tags.forEach(tag => allTags.add(tag)));
    } else if (activeTab === 'journal') {
      journalEntries.forEach(entry => entry.tags.forEach(tag => allTags.add(tag)));
    } else {
      reports.forEach(report => report.tags.forEach(tag => allTags.add(tag)));
    }
    return Array.from(allTags);
  };

  const getFilteredItems = () => {
    let items: any[] = [];
    if (activeTab === 'trades') items = trades;
    else if (activeTab === 'journal') items = journalEntries;
    else items = reports;

    return items.filter(item => {
      const matchesSearch = searchTerm === '' || 
        (activeTab === 'trades' && item.symbol.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (activeTab !== 'trades' && item.title.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesTag = filterTag === 'all' || item.tags.includes(filterTag);
      
      return matchesSearch && matchesTag;
    });
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const calculatePnL = (trade: TradeEntry) => {
    if (!trade.exitPrice) return 0;
    const multiplier = trade.type === 'long' ? 1 : -1;
    return (trade.exitPrice - trade.entryPrice) * trade.quantity * multiplier;
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Left Sidebar - Navigation and Filters */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Archive</h2>
          <p className="text-sm text-gray-400">Your Trading History & Insights</p>
        </div>

        {/* Tab Navigation */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex space-x-1">
            {[
              { id: 'trades', label: 'Trades', icon: '📈' },
              { id: 'journal', label: 'Journal', icon: '📝' },
              { id: 'reports', label: 'Reports', icon: '📊' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <span className="mr-1">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Search and Filters */}
        <div className="p-4 border-b border-gray-700 space-y-3">
          <input
            type="text"
            placeholder={`Search ${activeTab}...`}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          
          <select
            value={filterTag}
            onChange={(e) => setFilterTag(e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Tags</option>
            {getAllTags().map(tag => (
              <option key={tag} value={tag}>{tag}</option>
            ))}
          </select>

          <button
            onClick={() => setIsCreating(true)}
            className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
          >
            + New {activeTab.slice(0, -1)}
          </button>
        </div>

        {/* Items List */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-2">
            {getFilteredItems().map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedItem(item)}
                className={`p-3 rounded-lg cursor-pointer transition-colors border-l-4 ${
                  selectedItem?.id === item.id
                    ? 'bg-blue-600/20 border-blue-500'
                    : 'bg-gray-700 border-gray-600 hover:bg-gray-600'
                }`}
              >
                {activeTab === 'trades' ? (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-white">{item.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        item.type === 'long' ? 'bg-green-600' : 'bg-red-600'
                      }`}>
                        {item.type.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm text-gray-300">
                      Entry: ${item.entryPrice.toFixed(2)}
                      {item.exitPrice && ` → $${item.exitPrice.toFixed(2)}`}
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className={`text-xs px-2 py-1 rounded ${
                        item.status === 'open' ? 'bg-yellow-600' : 'bg-gray-600'
                      }`}>
                        {item.status}
                      </span>
                      {item.pnl && (
                        <span className={`text-sm font-medium ${
                          item.pnl > 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {item.pnl > 0 ? '+' : ''}${item.pnl.toFixed(2)}
                        </span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div>
                    <h4 className="font-semibold text-white text-sm mb-1">{item.title}</h4>
                    <p className="text-xs text-gray-400 mb-2">
                      {formatDate(activeTab === 'journal' ? item.date : item.createdDate)}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {item.tags.slice(0, 2).map((tag: string) => (
                        <span key={tag} className="bg-blue-600 text-white px-2 py-1 rounded text-xs">
                          {tag}
                        </span>
                      ))}
                      {item.tags.length > 2 && (
                        <span className="text-gray-400 text-xs">+{item.tags.length - 2}</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {selectedItem ? (
          <>
            {/* Header */}
            <div className="bg-gray-800 border-b border-gray-700 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white">
                    {activeTab === 'trades' ? (selectedItem as TradeEntry).symbol : (selectedItem as JournalEntry | Report).title}
                  </h3>
                  <p className="text-sm text-gray-400">
                    {formatDate(
                      activeTab === 'trades' ? (selectedItem as TradeEntry).entryDate :
                      activeTab === 'journal' ? (selectedItem as JournalEntry).date :
                      (selectedItem as Report).createdDate
                    )}
                  </p>
                </div>
                <div className="flex space-x-2">
                  <button className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors text-sm">
                    Edit
                  </button>
                  <button className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition-colors text-sm">
                    Delete
                  </button>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {activeTab === 'trades' && selectedItem && (
                <div className="space-y-6">
                  {/* Trade Summary */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Trade Summary</h4>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      <div>
                        <div className="text-sm text-gray-400">Symbol</div>
                        <div className="text-lg font-semibold text-white">{(selectedItem as TradeEntry).symbol}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-400">Type</div>
                        <div className={`text-lg font-semibold ${
                          (selectedItem as TradeEntry).type === 'long' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(selectedItem as TradeEntry).type.toUpperCase()}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-400">Quantity</div>
                        <div className="text-lg font-semibold text-white">{(selectedItem as TradeEntry).quantity}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-400">Status</div>
                        <div className={`text-lg font-semibold ${
                          (selectedItem as TradeEntry).status === 'open' ? 'text-yellow-400' : 'text-blue-400'
                        }`}>
                          {(selectedItem as TradeEntry).status.toUpperCase()}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Price Information */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Price Information</h4>
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                      <div>
                        <div className="text-sm text-gray-400">Entry Price</div>
                        <div className="text-xl font-semibold text-white">${(selectedItem as TradeEntry).entryPrice.toFixed(2)}</div>
                      </div>
                      {(selectedItem as TradeEntry).exitPrice && (
                        <div>
                          <div className="text-sm text-gray-400">Exit Price</div>
                          <div className="text-xl font-semibold text-white">${(selectedItem as TradeEntry).exitPrice!.toFixed(2)}</div>
                        </div>
                      )}
                      {(selectedItem as TradeEntry).pnl && (
                        <div>
                          <div className="text-sm text-gray-400">P&L</div>
                          <div className={`text-xl font-semibold ${
                            (selectedItem as TradeEntry).pnl! > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {(selectedItem as TradeEntry).pnl! > 0 ? '+' : ''}${(selectedItem as TradeEntry).pnl!.toFixed(2)}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Notes */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Notes</h4>
                    <p className="text-gray-300 leading-relaxed">{(selectedItem as TradeEntry).notes}</p>
                  </div>

                  {/* Tags */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Tags</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedItem.tags.map((tag) => (
                        <span key={tag} className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'journal' && (
                <div className="space-y-6">
                  {/* Journal Header */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-lg font-semibold text-white">{(selectedItem as JournalEntry).title}</h4>
                      <span className={`px-3 py-1 rounded-full text-sm ${
                        (selectedItem as JournalEntry).mood === 'bullish' ? 'bg-green-600' :
                        (selectedItem as JournalEntry).mood === 'bearish' ? 'bg-red-600' : 'bg-gray-600'
                      }`}>
                        {(selectedItem as JournalEntry).mood}
                      </span>
                    </div>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-gray-400">Market Conditions</div>
                        <div className="text-white">{(selectedItem as JournalEntry).marketConditions}</div>
                      </div>
                      <div>
                        <div className="text-gray-400">Key Lessons</div>
                        <div className="text-white">{(selectedItem as JournalEntry).lessons}</div>
                      </div>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Content</h4>
                    <div className="prose prose-invert max-w-none">
                      <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">{(selectedItem as JournalEntry).content}</p>
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Tags</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedItem.tags.map((tag) => (
                        <span key={tag} className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'reports' && (
                <div className="space-y-6">
                  {/* Report Header */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-lg font-semibold text-white">{(selectedItem as Report).title}</h4>
                      <span className={`px-3 py-1 rounded-full text-sm ${
                        (selectedItem as Report).type === 'analysis' ? 'bg-blue-600' :
                        (selectedItem as Report).type === 'strategy' ? 'bg-green-600' : 'bg-purple-600'
                      }`}>
                        {(selectedItem as Report).type}
                      </span>
                    </div>
                    <div className="text-sm text-gray-400">
                      Created: {formatDate((selectedItem as Report).createdDate)}
                      {(selectedItem as Report).lastModified !== (selectedItem as Report).createdDate && (
                        <span className="ml-4">
                          Last Modified: {formatDate((selectedItem as Report).lastModified)}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Content */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Content</h4>
                    <div className="prose prose-invert max-w-none">
                      <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">{(selectedItem as Report).content}</p>
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="bg-gray-800 rounded-lg p-4">
                    <h4 className="text-lg font-semibold mb-4 text-white">Tags</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedItem.tags.map((tag) => (
                        <span key={tag} className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p>Select an item from the archive to view details</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}