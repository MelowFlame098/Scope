'use client'

import { useState, useEffect } from 'react'
import {
  MagnifyingGlassIcon,
  ChartBarIcon,
  DocumentTextIcon,
  CurrencyDollarIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  InformationCircleIcon,
  BookmarkIcon,
  ShareIcon,
  CalendarIcon
} from '@heroicons/react/24/outline'

interface AssetData {
  id: string
  name: string
  symbol: string
  price: number
  change24h: number
  changePercent24h: number
  volume24h: number
  marketCap?: number
  category: 'crypto' | 'stocks' | 'forex' | 'commodities'
  description: string
  fundamentals: {
    pe?: number
    eps?: number
    dividend?: number
    beta?: number
    marketCap?: number
    volume?: number
    supply?: number
    holders?: number
  }
  technicals: {
    rsi: number
    macd: number
    sma20: number
    sma50: number
    support: number
    resistance: number
  }
  news: number
  sentiment: 'bullish' | 'bearish' | 'neutral'
  rating: number
}

interface ResearchReport {
  id: string
  title: string
  summary: string
  author: string
  publishedAt: string
  rating: 'buy' | 'hold' | 'sell'
  targetPrice?: number
  confidence: number
}

const mockAssets: AssetData[] = [
  {
    id: 'bitcoin',
    name: 'Bitcoin',
    symbol: 'BTC',
    price: 43250.50,
    change24h: 1250.30,
    changePercent24h: 2.98,
    volume24h: 28500000000,
    marketCap: 847000000000,
    category: 'crypto',
    description: 'Bitcoin is a decentralized digital currency that can be transferred on the peer-to-peer bitcoin network.',
    fundamentals: {
      marketCap: 847000000000,
      volume: 28500000000,
      supply: 19500000,
      holders: 106000000
    },
    technicals: {
      rsi: 68.5,
      macd: 245.8,
      sma20: 42100,
      sma50: 40800,
      support: 41500,
      resistance: 45000
    },
    news: 156,
    sentiment: 'bullish',
    rating: 4.2
  },
  {
    id: 'apple',
    name: 'Apple Inc.',
    symbol: 'AAPL',
    price: 185.92,
    change24h: 2.15,
    changePercent24h: 1.17,
    volume24h: 45000000,
    marketCap: 2890000000000,
    category: 'stocks',
    description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.',
    fundamentals: {
      pe: 28.5,
      eps: 6.52,
      dividend: 0.96,
      beta: 1.25,
      marketCap: 2890000000000,
      volume: 45000000
    },
    technicals: {
      rsi: 55.2,
      macd: 1.85,
      sma20: 183.50,
      sma50: 178.20,
      support: 180.00,
      resistance: 190.00
    },
    news: 89,
    sentiment: 'bullish',
    rating: 4.0
  },
  {
    id: 'tesla',
    name: 'Tesla Inc.',
    symbol: 'TSLA',
    price: 248.50,
    change24h: -12.30,
    changePercent24h: -4.72,
    volume24h: 125000000,
    marketCap: 789000000000,
    category: 'stocks',
    description: 'Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems.',
    fundamentals: {
      pe: 65.8,
      eps: 3.78,
      dividend: 0,
      beta: 2.05,
      marketCap: 789000000000,
      volume: 125000000
    },
    technicals: {
      rsi: 35.8,
      macd: -5.25,
      sma20: 255.80,
      sma50: 268.40,
      support: 240.00,
      resistance: 270.00
    },
    news: 67,
    sentiment: 'bearish',
    rating: 3.2
  }
]

const mockReports: ResearchReport[] = [
  {
    id: '1',
    title: 'Bitcoin: Institutional Adoption Driving Long-term Growth',
    summary: 'Strong institutional demand and improving regulatory clarity support bullish outlook for Bitcoin.',
    author: 'FinScope Research',
    publishedAt: '2024-01-15T10:00:00Z',
    rating: 'buy',
    targetPrice: 50000,
    confidence: 85
  },
  {
    id: '2',
    title: 'Apple: Services Growth Offsetting Hardware Challenges',
    summary: 'Despite iPhone sales pressure, services revenue growth maintains positive outlook.',
    author: 'Tech Analysis Pro',
    publishedAt: '2024-01-15T09:30:00Z',
    rating: 'hold',
    targetPrice: 190,
    confidence: 72
  }
]

export default function AssetResearch() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedAsset, setSelectedAsset] = useState<AssetData | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'fundamentals' | 'technicals' | 'reports'>('overview')
  const [filteredAssets, setFilteredAssets] = useState<AssetData[]>(mockAssets)
  const [reports, setReports] = useState<ResearchReport[]>(mockReports)
  const [isClient, setIsClient] = useState(false)

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (searchQuery) {
      setFilteredAssets(
        mockAssets.filter(asset => 
          asset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          asset.symbol.toLowerCase().includes(searchQuery.toLowerCase())
        )
      )
    } else {
      setFilteredAssets(mockAssets)
    }
  }, [searchQuery])

  const formatNumber = (num: number, decimals: number = 2): string => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(decimals)}T`
    if (num >= 1e9) return `$${(num / 1e9).toFixed(decimals)}B`
    if (num >= 1e6) return `$${(num / 1e6).toFixed(decimals)}M`
    if (num >= 1e3) return `$${(num / 1e3).toFixed(decimals)}K`
    return `$${num.toFixed(decimals)}`
  }

  const getRatingColor = (rating: number): string => {
    if (rating >= 4) return 'text-green-400'
    if (rating >= 3) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getSentimentColor = (sentiment: string): string => {
    switch (sentiment) {
      case 'bullish': return 'text-green-400'
      case 'bearish': return 'text-red-400'
      default: return 'text-yellow-400'
    }
  }

  const getRSIColor = (rsi: number): string => {
    if (rsi >= 70) return 'text-red-400' // Overbought
    if (rsi <= 30) return 'text-green-400' // Oversold
    return 'text-yellow-400' // Neutral
  }

  const tabs = [
    { id: 'overview', name: 'Overview', icon: InformationCircleIcon },
    { id: 'fundamentals', name: 'Fundamentals', icon: DocumentTextIcon },
    { id: 'technicals', name: 'Technicals', icon: ChartBarIcon },
    { id: 'reports', name: 'Reports', icon: BookmarkIcon }
  ]

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="card">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center">
          <MagnifyingGlassIcon className="h-6 w-6 mr-2" />
          Asset Research
        </h2>
        
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-dark-400" />
          <input
            type="text"
            placeholder="Search assets (e.g., BTC, AAPL, TSLA)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-field pl-10"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Asset List */}
        <div className="lg:col-span-1">
          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4">Assets</h3>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredAssets.map(asset => {
                const isSelected = selectedAsset?.id === asset.id
                const isPositive = asset.changePercent24h >= 0
                
                return (
                  <div
                    key={asset.id}
                    onClick={() => setSelectedAsset(asset)}
                    className={`p-3 rounded border cursor-pointer transition-all ${
                      isSelected 
                        ? 'border-primary-500 bg-primary-900/20' 
                        : 'border-dark-600 hover:border-dark-500 hover:bg-dark-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-white font-medium">{asset.name}</p>
                        <p className="text-dark-400 text-sm">{asset.symbol}</p>
                      </div>
                      
                      <div className="text-right">
                        <p className="text-white font-medium">
                          {formatNumber(asset.price, 2)}
                        </p>
                        <div className={`flex items-center justify-end space-x-1 ${
                          isPositive ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {isPositive ? (
                            <ArrowUpIcon className="h-3 w-3" />
                ) : (
                  <ArrowDownIcon className="h-3 w-3" />
                          )}
                          <span className="text-xs">
                            {Math.abs(asset.changePercent24h).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between mt-2">
                      <div className="flex items-center space-x-2">
                        <span className={`w-2 h-2 rounded-full ${
                          asset.sentiment === 'bullish' ? 'bg-green-400' :
                          asset.sentiment === 'bearish' ? 'bg-red-400' : 'bg-yellow-400'
                        }`}></span>
                        <span className="text-xs text-dark-400 capitalize">{asset.sentiment}</span>
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        <span className={`text-xs ${getRatingColor(asset.rating)}`}>
                          ★ {asset.rating.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Asset Details */}
        <div className="lg:col-span-2">
          {selectedAsset ? (
            <div className="space-y-6">
              {/* Asset Header */}
              <div className="card">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-primary-600 rounded-full flex items-center justify-center">
                      <CurrencyDollarIcon className="h-6 w-6 text-white" />
                    </div>
                    
                    <div>
                      <h2 className="text-xl font-bold text-white">{selectedAsset.name}</h2>
                      <p className="text-dark-400">{selectedAsset.symbol}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button className="p-2 bg-dark-700 hover:bg-dark-600 rounded transition-colors">
                      <BookmarkIcon className="h-4 w-4 text-dark-300" />
                    </button>
                    <button className="p-2 bg-dark-700 hover:bg-dark-600 rounded transition-colors">
                      <ShareIcon className="h-4 w-4 text-dark-300" />
                    </button>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                  <div>
                    <p className="text-dark-400 text-sm">Price</p>
                    <p className="text-white text-lg font-semibold">
                      {formatNumber(selectedAsset.price, 2)}
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-dark-400 text-sm">24h Change</p>
                    <p className={`text-lg font-semibold ${
                      selectedAsset.changePercent24h >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {selectedAsset.changePercent24h >= 0 ? '+' : ''}{selectedAsset.changePercent24h.toFixed(2)}%
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-dark-400 text-sm">Volume</p>
                    <p className="text-white text-lg font-semibold">
                      {formatNumber(selectedAsset.volume24h)}
                    </p>
                  </div>
                  
                  <div>
                    <p className="text-dark-400 text-sm">Rating</p>
                    <p className={`text-lg font-semibold ${getRatingColor(selectedAsset.rating)}`}>
                      ★ {selectedAsset.rating.toFixed(1)}
                    </p>
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div className="card">
                <div className="flex space-x-1 mb-6">
                  {tabs.map(tab => {
                    const IconComponent = tab.icon
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as any)}
                        className={`flex items-center space-x-2 px-4 py-2 rounded transition-colors ${
                          activeTab === tab.id
                            ? 'bg-primary-600 text-white'
                            : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                        }`}
                      >
                        <IconComponent className="h-4 w-4" />
                        <span>{tab.name}</span>
                      </button>
                    )
                  })}
                </div>

                {/* Tab Content */}
                {activeTab === 'overview' && (
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-white font-medium mb-2">Description</h4>
                      <p className="text-dark-300 text-sm leading-relaxed">
                        {selectedAsset.description}
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-4 bg-dark-700 rounded">
                        <h5 className="text-white font-medium mb-2">Market Sentiment</h5>
                        <div className="flex items-center space-x-2">
                          <span className={`w-3 h-3 rounded-full ${
                            selectedAsset.sentiment === 'bullish' ? 'bg-green-400' :
                            selectedAsset.sentiment === 'bearish' ? 'bg-red-400' : 'bg-yellow-400'
                          }`}></span>
                          <span className={`capitalize ${getSentimentColor(selectedAsset.sentiment)}`}>
                            {selectedAsset.sentiment}
                          </span>
                        </div>
                      </div>
                      
                      <div className="p-4 bg-dark-700 rounded">
                        <h5 className="text-white font-medium mb-2">News Coverage</h5>
                        <p className="text-white">{selectedAsset.news} articles</p>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'fundamentals' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(selectedAsset.fundamentals).map(([key, value]) => {
                      if (value === undefined) return null
                      
                      let displayValue: string | number = value
                      if ((key === 'marketCap' || key === 'volume') && typeof value === 'number') {
                        displayValue = formatNumber(value)
                      } else if (typeof value === 'number') {
                        displayValue = value.toLocaleString()
                      }
                      
                      return (
                        <div key={key} className="p-4 bg-dark-700 rounded">
                          <p className="text-dark-400 text-sm capitalize">
                            {key.replace(/([A-Z])/g, ' $1').trim()}
                          </p>
                          <p className="text-white text-lg font-semibold">
                            {displayValue}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                )}

                {activeTab === 'technicals' && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-dark-700 rounded">
                      <p className="text-dark-400 text-sm">RSI (14)</p>
                      <p className={`text-lg font-semibold ${getRSIColor(selectedAsset.technicals.rsi)}`}>
                        {selectedAsset.technicals.rsi.toFixed(1)}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-dark-700 rounded">
                      <p className="text-dark-400 text-sm">MACD</p>
                      <p className={`text-lg font-semibold ${
                        selectedAsset.technicals.macd >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {selectedAsset.technicals.macd.toFixed(2)}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-dark-700 rounded">
                      <p className="text-dark-400 text-sm">SMA 20</p>
                      <p className="text-white text-lg font-semibold">
                        {formatNumber(selectedAsset.technicals.sma20, 2)}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-dark-700 rounded">
                      <p className="text-dark-400 text-sm">SMA 50</p>
                      <p className="text-white text-lg font-semibold">
                        {formatNumber(selectedAsset.technicals.sma50, 2)}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-dark-700 rounded">
                      <p className="text-dark-400 text-sm">Support</p>
                      <p className="text-green-400 text-lg font-semibold">
                        {formatNumber(selectedAsset.technicals.support, 2)}
                      </p>
                    </div>
                    
                    <div className="p-4 bg-dark-700 rounded">
                      <p className="text-dark-400 text-sm">Resistance</p>
                      <p className="text-red-400 text-lg font-semibold">
                        {formatNumber(selectedAsset.technicals.resistance, 2)}
                      </p>
                    </div>
                  </div>
                )}

                {activeTab === 'reports' && (
                  <div className="space-y-4">
                    {reports.filter(report => 
                      report.title.toLowerCase().includes(selectedAsset.name.toLowerCase())
                    ).map(report => (
                      <div key={report.id} className="p-4 bg-dark-700 rounded">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h5 className="text-white font-medium mb-1">{report.title}</h5>
                            <p className="text-dark-300 text-sm mb-2">{report.summary}</p>
                            
                            <div className="flex items-center space-x-4 text-xs text-dark-400">
                              <div className="flex items-center space-x-1">
                                <CalendarIcon className="h-3 w-3" />
                                <span>{isClient ? new Date(report.publishedAt).toLocaleDateString() : ''}</span>
                              </div>
                              <span>by {report.author}</span>
                            </div>
                          </div>
                          
                          <div className="text-right ml-4">
                            <span className={`px-2 py-1 text-xs rounded ${
                              report.rating === 'buy' ? 'bg-green-900/30 text-green-400' :
                              report.rating === 'sell' ? 'bg-red-900/30 text-red-400' :
                              'bg-yellow-900/30 text-yellow-400'
                            }`}>
                              {report.rating.toUpperCase()}
                            </span>
                            
                            {report.targetPrice && (
                              <p className="text-white text-sm mt-1">
                                Target: {formatNumber(report.targetPrice, 2)}
                              </p>
                            )}
                            
                            <p className="text-dark-400 text-xs">
                              {report.confidence}% confidence
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                    
                    {reports.filter(report => 
                      report.title.toLowerCase().includes(selectedAsset.name.toLowerCase())
                    ).length === 0 && (
                      <div className="text-center py-8">
                        <DocumentTextIcon className="h-12 w-12 text-dark-500 mx-auto mb-2" />
                        <p className="text-dark-400">No research reports available</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="card text-center py-12">
              <MagnifyingGlassIcon className="h-16 w-16 text-dark-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">Select an Asset</h3>
              <p className="text-dark-400">Choose an asset from the list to view detailed research and analysis</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}