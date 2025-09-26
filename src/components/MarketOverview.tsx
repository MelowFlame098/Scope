'use client'

import { useState, useEffect } from 'react'
import {
  ArrowUpIcon,
  ArrowDownIcon,
  GlobeAltIcon,
  CurrencyDollarIcon,
  ChartBarSquareIcon,
  FireIcon
} from '@heroicons/react/24/outline'

interface MarketData {
  id: string
  name: string
  symbol: string
  price: number
  change24h: number
  changePercent24h: number
  volume24h: number
  marketCap?: number
  category: 'crypto' | 'stocks' | 'forex' | 'commodities'
}

interface MarketStats {
  totalMarketCap: number
  totalVolume: number
  btcDominance: number
  fearGreedIndex: number
  activeAssets: number
}

const mockMarketData: MarketData[] = [
  {
    id: 'bitcoin',
    name: 'Bitcoin',
    symbol: 'BTC',
    price: 43250.50,
    change24h: 1250.30,
    changePercent24h: 2.98,
    volume24h: 28500000000,
    marketCap: 847000000000,
    category: 'crypto'
  },
  {
    id: 'ethereum',
    name: 'Ethereum',
    symbol: 'ETH',
    price: 2650.75,
    change24h: -85.25,
    changePercent24h: -3.11,
    volume24h: 15200000000,
    marketCap: 318000000000,
    category: 'crypto'
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
    category: 'stocks'
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
    category: 'stocks'
  },
  {
    id: 'eurusd',
    name: 'EUR/USD',
    symbol: 'EURUSD',
    price: 1.0875,
    change24h: 0.0025,
    changePercent24h: 0.23,
    volume24h: 1200000000,
    category: 'forex'
  },
  {
    id: 'gold',
    name: 'Gold',
    symbol: 'XAU',
    price: 2045.80,
    change24h: 15.60,
    changePercent24h: 0.77,
    volume24h: 850000000,
    category: 'commodities'
  }
]

const mockMarketStats: MarketStats = {
  totalMarketCap: 1650000000000,
  totalVolume: 89500000000,
  btcDominance: 51.3,
  fearGreedIndex: 72,
  activeAssets: 2847
}

const categories = {
  crypto: { name: 'Crypto', icon: CurrencyDollarIcon, color: 'text-orange-400' },
  stocks: { name: 'Stocks', icon: ChartBarSquareIcon, color: 'text-blue-400' },
  forex: { name: 'Forex', icon: GlobeAltIcon, color: 'text-green-400' },
  commodities: { name: 'Commodities', icon: FireIcon, color: 'text-yellow-400' }
}

export default function MarketOverview() {
  const [marketData, setMarketData] = useState<MarketData[]>(mockMarketData)
  const [marketStats, setMarketStats] = useState<MarketStats>(mockMarketStats)
  const [activeCategory, setActiveCategory] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'marketCap' | 'volume' | 'change'>('marketCap')

  const filteredData = marketData.filter(asset => 
    activeCategory === 'all' || asset.category === activeCategory
  )

  const sortedData = [...filteredData].sort((a, b) => {
    switch (sortBy) {
      case 'marketCap':
        return (b.marketCap || 0) - (a.marketCap || 0)
      case 'volume':
        return b.volume24h - a.volume24h
      case 'change':
        return b.changePercent24h - a.changePercent24h
      default:
        return 0
    }
  })

  const formatNumber = (num: number, decimals: number = 2): string => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(decimals)}T`
    if (num >= 1e9) return `$${(num / 1e9).toFixed(decimals)}B`
    if (num >= 1e6) return `$${(num / 1e6).toFixed(decimals)}M`
    if (num >= 1e3) return `$${(num / 1e3).toFixed(decimals)}K`
    return `$${num.toFixed(decimals)}`
  }

  const getFearGreedColor = (index: number): string => {
    if (index >= 75) return 'text-red-400'
    if (index >= 55) return 'text-yellow-400'
    if (index >= 45) return 'text-green-400'
    if (index >= 25) return 'text-orange-400'
    return 'text-red-400'
  }

  const getFearGreedLabel = (index: number): string => {
    if (index >= 75) return 'Extreme Greed'
    if (index >= 55) return 'Greed'
    if (index >= 45) return 'Neutral'
    if (index >= 25) return 'Fear'
    return 'Extreme Fear'
  }

  return (
    <div className="space-y-6">
      {/* Market Stats */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <GlobeAltIcon className="h-5 w-5 mr-2" />
          Market Overview
        </h3>
        
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="text-center">
            <p className="text-xs text-dark-400">Total Market Cap</p>
            <p className="text-lg font-semibold text-white">
              {formatNumber(marketStats.totalMarketCap)}
            </p>
          </div>
          
          <div className="text-center">
            <p className="text-xs text-dark-400">24h Volume</p>
            <p className="text-lg font-semibold text-white">
              {formatNumber(marketStats.totalVolume)}
            </p>
          </div>
          
          <div className="text-center">
            <p className="text-xs text-dark-400">BTC Dominance</p>
            <p className="text-lg font-semibold text-white">
              {marketStats.btcDominance}%
            </p>
          </div>
          
          <div className="text-center">
            <p className="text-xs text-dark-400">Fear & Greed</p>
            <p className={`text-lg font-semibold ${getFearGreedColor(marketStats.fearGreedIndex)}`}>
              {marketStats.fearGreedIndex}
            </p>
            <p className="text-xs text-dark-500">
              {getFearGreedLabel(marketStats.fearGreedIndex)}
            </p>
          </div>
          
          <div className="text-center">
            <p className="text-xs text-dark-400">Active Assets</p>
            <p className="text-lg font-semibold text-white">
              {marketStats.activeAssets.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Top Movers */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Top Assets</h3>
          
          <div className="flex items-center space-x-2">
            {/* Category Filter */}
            <select
              value={activeCategory}
              onChange={(e) => setActiveCategory(e.target.value)}
              className="bg-dark-700 border border-dark-600 text-white text-sm rounded px-2 py-1"
            >
              <option value="all">All Categories</option>
              {Object.entries(categories).map(([key, cat]) => (
                <option key={key} value={key}>{cat.name}</option>
              ))}
            </select>
            
            {/* Sort Filter */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-dark-700 border border-dark-600 text-white text-sm rounded px-2 py-1"
            >
              <option value="marketCap">Market Cap</option>
              <option value="volume">Volume</option>
              <option value="change">24h Change</option>
            </select>
          </div>
        </div>

        <div className="space-y-2">
          {sortedData.slice(0, 10).map((asset, index) => {
            const CategoryIcon = categories[asset.category].icon
            const isPositive = asset.changePercent24h >= 0
            
            return (
              <div
                key={asset.id}
                className="flex items-center justify-between p-3 bg-dark-700 rounded hover:bg-dark-600 transition-colors cursor-pointer"
              >
                <div className="flex items-center space-x-3">
                  <span className="text-dark-400 text-sm w-6">{index + 1}</span>
                  
                  <div className="flex items-center space-x-2">
                    <CategoryIcon className={`h-4 w-4 ${categories[asset.category].color}`} />
                    <div>
                      <p className="text-white font-medium">{asset.name}</p>
                      <p className="text-dark-400 text-xs">{asset.symbol}</p>
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <p className="text-white font-medium">
                    {asset.category === 'forex' ? asset.price.toFixed(4) : formatNumber(asset.price, 2)}
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

                <div className="text-right">
                  <p className="text-dark-300 text-xs">Volume</p>
                  <p className="text-white text-sm">
                    {formatNumber(asset.volume24h)}
                  </p>
                </div>

                {asset.marketCap && (
                  <div className="text-right">
                    <p className="text-dark-300 text-xs">Market Cap</p>
                    <p className="text-white text-sm">
                      {formatNumber(asset.marketCap)}
                    </p>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Market Sentiment */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Market Sentiment</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-dark-700 rounded">
            <div className={`text-2xl font-bold ${getFearGreedColor(marketStats.fearGreedIndex)}`}>
              {marketStats.fearGreedIndex}
            </div>
            <p className="text-white font-medium">Fear & Greed Index</p>
            <p className="text-dark-400 text-sm">{getFearGreedLabel(marketStats.fearGreedIndex)}</p>
          </div>
          
          <div className="text-center p-4 bg-dark-700 rounded">
            <div className="text-2xl font-bold text-green-400">68%</div>
            <p className="text-white font-medium">Bullish Sentiment</p>
            <p className="text-dark-400 text-sm">Social Media</p>
          </div>
          
          <div className="text-center p-4 bg-dark-700 rounded">
            <div className="text-2xl font-bold text-blue-400">7.2</div>
            <p className="text-white font-medium">News Sentiment</p>
            <p className="text-dark-400 text-sm">Positive Bias</p>
          </div>
        </div>
      </div>
    </div>
  )
}