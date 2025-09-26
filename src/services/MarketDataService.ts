import { EventEmitter } from 'events';
import { redisService } from './RedisService';

export interface PriceData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  marketCap?: number;
  timestamp: number;
}

export interface CandlestickData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketStats {
  totalMarketCap: number;
  totalVolume24h: number;
  btcDominance: number;
  activeSymbols: number;
  topGainers: PriceData[];
  topLosers: PriceData[];
  mostActive: PriceData[];
}

export interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'buy' | 'sell' | 'neutral';
  timestamp: number;
}

export type TimeFrame = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';

class MarketDataService extends EventEmitter {
  private priceCache: Map<string, PriceData> = new Map();
  private wsConnection: WebSocket | null = null;
  private subscriptions: Set<string> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  // Popular trading symbols
  private symbols = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD', 'MATIC-USD',
    'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'VIX'
  ];

  constructor() {
    super();
    this.initializeWebSocket();
    this.startPriceUpdates();
  }

  private initializeWebSocket() {
    try {
      // Connect to our backend WebSocket endpoint
      const wsUrl = process.env.NODE_ENV === 'production' 
        ? 'wss://api.finscope.com/ws'
        : 'ws://localhost:8000/ws';
      
      this.wsConnection = new WebSocket(wsUrl);
      
      this.wsConnection.onopen = () => {
        console.log('Market data WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected');
        
        // Subscribe to market data updates
        this.sendWebSocketMessage({
          type: 'subscribe',
          symbols: Array.from(this.subscriptions)
        });
      };
      
      this.wsConnection.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.wsConnection.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        this.emit('disconnected');
        this.scheduleReconnect();
      };
      
      this.wsConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.scheduleReconnect();
    }
  }

  private handleWebSocketMessage(message: any) {
    switch (message.type) {
      case 'price_update':
        this.handlePriceUpdate(message.data);
        break;
      case 'market_update':
        this.handleMarketUpdate(message.data);
        break;
      case 'status':
        console.log('WebSocket status:', message.data);
        break;
      case 'error':
        console.error('WebSocket error message:', message.data);
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  }

  private handlePriceUpdate(data: any) {
    const priceData: PriceData = {
      symbol: data.symbol,
      price: data.price,
      change: data.change,
      changePercent: data.change_percent,
      volume: data.volume || 0,
      high24h: data.high_24h || data.price * 1.05,
      low24h: data.low_24h || data.price * 0.95,
      marketCap: data.market_cap,
      timestamp: new Date(data.timestamp).getTime()
    };
    
    this.priceCache.set(data.symbol, priceData);
    this.emit('priceUpdate', { symbol: data.symbol, data: priceData });
  }

  private handleMarketUpdate(data: any) {
    // Handle bulk market data updates
    if (data.prices) {
      Object.entries(data.prices).forEach(([symbol, priceInfo]: [string, any]) => {
        this.handlePriceUpdate({ symbol, ...priceInfo });
      });
    }
  }

  private sendWebSocketMessage(message: any) {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify(message));
    }
  }

  private startPriceUpdates() {
    // Simulate real-time price updates
    setInterval(() => {
      this.symbols.forEach(symbol => {
        const priceData = this.generateMockPriceData(symbol);
        this.updatePrice(symbol, priceData);
      });
    }, 2000); // Update every 2 seconds
  }

  private async updatePrice(symbol: string, priceData: PriceData) {
    // Cache in memory for immediate access
    this.priceCache.set(symbol, priceData);
    
    // Cache in Redis for persistence and sharing
    await redisService.cacheMarketData(symbol, priceData, 60);
    
    // Publish real-time update
    await redisService.publish(`price:${symbol}`, priceData);
    
    // Emit to local subscribers
    this.emit('priceUpdate', { symbol, data: priceData });
  }

  private generateMockPriceData(symbol: string): PriceData {
    const basePrice = this.getBasePrice(symbol);
    const volatility = this.getVolatility(symbol);
    
    // Generate realistic price movement
    const changePercent = (Math.random() - 0.5) * volatility;
    const change = basePrice * (changePercent / 100);
    const price = basePrice + change;
    
    const volume = Math.floor(Math.random() * 10000000) + 1000000;
    const high24h = price * (1 + Math.random() * 0.05);
    const low24h = price * (1 - Math.random() * 0.05);
    
    return {
      symbol,
      price: Number(price.toFixed(2)),
      change: Number(change.toFixed(2)),
      changePercent: Number(changePercent.toFixed(2)),
      volume,
      high24h: Number(high24h.toFixed(2)),
      low24h: Number(low24h.toFixed(2)),
      marketCap: this.calculateMarketCap(symbol, price),
      timestamp: Date.now()
    };
  }

  private getBasePrice(symbol: string): number {
    const prices: Record<string, number> = {
      'AAPL': 175.50,
      'GOOGL': 2850.00,
      'MSFT': 415.25,
      'AMZN': 3200.00,
      'TSLA': 245.80,
      'META': 485.60,
      'NVDA': 875.30,
      'NFLX': 425.90,
      'BTC-USD': 43500.00,
      'ETH-USD': 2650.00,
      'ADA-USD': 0.48,
      'SOL-USD': 98.50,
      'DOT-USD': 7.25,
      'MATIC-USD': 0.85,
      'SPY': 485.20,
      'QQQ': 395.80,
      'IWM': 198.50,
      'GLD': 185.30,
      'SLV': 22.45,
      'VIX': 15.80
    };
    return prices[symbol] || 100.00;
  }

  private getVolatility(symbol: string): number {
    // Different asset classes have different volatility
    if (symbol.includes('BTC') || symbol.includes('ETH')) return 8.0; // Crypto high volatility
    if (symbol.includes('USD')) return 12.0; // Other crypto
    if (symbol === 'VIX') return 15.0; // VIX is very volatile
    if (symbol === 'TSLA') return 6.0; // Tesla high volatility
    return 3.0; // Regular stocks
  }

  private calculateMarketCap(symbol: string, price: number): number | undefined {
    const shares: Record<string, number> = {
      'AAPL': 15728000000,
      'GOOGL': 12800000000,
      'MSFT': 7430000000,
      'AMZN': 10700000000,
      'TSLA': 3170000000,
      'META': 2650000000,
      'NVDA': 2470000000,
      'NFLX': 442000000
    };
    
    const shareCount = shares[symbol];
    return shareCount ? Math.floor(shareCount * price) : undefined;
  }

  // Public API Methods
  async getCurrentPrice(symbol: string): Promise<PriceData | null> {
    // Try Redis cache first
    const cachedData = await redisService.getMarketData(symbol);
    if (cachedData) {
      return cachedData;
    }

    // Try memory cache
    const memoryData = this.priceCache.get(symbol);
    if (memoryData) {
      return memoryData;
    }

    // Generate fresh data if not cached
    const priceData = this.generateMockPriceData(symbol);
    await this.updatePrice(symbol, priceData);
    return priceData;
  }

  async getMultiplePrices(symbols: string[]): Promise<Record<string, PriceData>> {
    const prices: Record<string, PriceData> = {};
    
    await Promise.all(
      symbols.map(async (symbol) => {
        const price = await this.getCurrentPrice(symbol);
        if (price) {
          prices[symbol] = price;
        }
      })
    );
    
    return prices;
  }

  async getHistoricalData(symbol: string, timeframe: TimeFrame, limit: number = 100): Promise<CandlestickData[]> {
    const cacheKey = `historical:${symbol}:${timeframe}:${limit}`;
    const cachedData = await redisService.get<CandlestickData[]>(cacheKey);
    
    if (cachedData) {
      return cachedData;
    }

    // Generate mock historical data
    const data = this.generateMockHistoricalData(symbol, timeframe, limit);
    
    // Cache for 5 minutes
    await redisService.set(cacheKey, data, 300);
    
    return data;
  }

  private generateMockHistoricalData(symbol: string, timeframe: TimeFrame, limit: number): CandlestickData[] {
    const basePrice = this.getBasePrice(symbol);
    const volatility = this.getVolatility(symbol);
    const data: CandlestickData[] = [];
    
    const timeframeMs = this.getTimeframeMs(timeframe);
    const now = Date.now();
    
    let currentPrice = basePrice;
    
    for (let i = limit - 1; i >= 0; i--) {
      const timestamp = now - (i * timeframeMs);
      
      // Generate OHLC data
      const open = currentPrice;
      const changePercent = (Math.random() - 0.5) * (volatility / 10);
      const close = open * (1 + changePercent / 100);
      
      const high = Math.max(open, close) * (1 + Math.random() * 0.02);
      const low = Math.min(open, close) * (1 - Math.random() * 0.02);
      const volume = Math.floor(Math.random() * 5000000) + 500000;
      
      data.push({
        timestamp,
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume
      });
      
      currentPrice = close;
    }
    
    return data;
  }

  private getTimeframeMs(timeframe: TimeFrame): number {
    const timeframes: Record<TimeFrame, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000,
      '1w': 7 * 24 * 60 * 60 * 1000,
      '1M': 30 * 24 * 60 * 60 * 1000
    };
    return timeframes[timeframe];
  }

  async getMarketStats(): Promise<MarketStats> {
    const cacheKey = 'market:stats';
    const cachedStats = await redisService.get<MarketStats>(cacheKey);
    
    if (cachedStats) {
      return cachedStats;
    }

    // Generate fresh market statistics
    const allPrices = await this.getMultiplePrices(this.symbols);
    const priceArray = Object.values(allPrices);
    
    // Sort by change percentage
    const sortedByChange = [...priceArray].sort((a, b) => b.changePercent - a.changePercent);
    const sortedByVolume = [...priceArray].sort((a, b) => b.volume - a.volume);
    
    const stats: MarketStats = {
      totalMarketCap: priceArray.reduce((sum, p) => sum + (p.marketCap || 0), 0),
      totalVolume24h: priceArray.reduce((sum, p) => sum + p.volume, 0),
      btcDominance: 42.5, // Mock BTC dominance
      activeSymbols: this.symbols.length,
      topGainers: sortedByChange.slice(0, 5),
      topLosers: sortedByChange.slice(-5).reverse(),
      mostActive: sortedByVolume.slice(0, 5)
    };
    
    // Cache for 1 minute
    await redisService.set(cacheKey, stats, 60);
    
    return stats;
  }

  async getTechnicalIndicators(symbol: string): Promise<TechnicalIndicator[]> {
    const cacheKey = `indicators:${symbol}`;
    const cachedIndicators = await redisService.get<TechnicalIndicator[]>(cacheKey);
    
    if (cachedIndicators) {
      return cachedIndicators;
    }

    // Generate mock technical indicators
    const indicators: TechnicalIndicator[] = [
      {
        name: 'RSI',
        value: Math.random() * 100,
        signal: Math.random() > 0.5 ? 'buy' : Math.random() > 0.5 ? 'sell' : 'neutral',
        timestamp: Date.now()
      },
      {
        name: 'MACD',
        value: (Math.random() - 0.5) * 10,
        signal: Math.random() > 0.5 ? 'buy' : Math.random() > 0.5 ? 'sell' : 'neutral',
        timestamp: Date.now()
      },
      {
        name: 'SMA_20',
        value: this.getBasePrice(symbol) * (0.95 + Math.random() * 0.1),
        signal: 'neutral',
        timestamp: Date.now()
      }
    ];
    
    // Cache for 2 minutes
    await redisService.set(cacheKey, indicators, 120);
    
    return indicators;
  }

  // Subscription Management
  subscribeToSymbol(symbol: string): void {
    this.subscriptions.add(symbol);
    
    // Send subscription message via WebSocket if connected
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.sendWebSocketMessage({
        type: 'subscribe',
        symbols: [symbol]
      });
    }
    
    // Subscribe to Redis pub/sub for real-time updates
    redisService.subscribe(`price:${symbol}`, (data: PriceData) => {
      this.emit('priceUpdate', { symbol, data });
    });
  }

  async unsubscribeFromSymbol(symbol: string): Promise<void> {
    this.subscriptions.delete(symbol);
    
    // Send unsubscription message via WebSocket if connected
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.sendWebSocketMessage({
        type: 'unsubscribe',
        symbols: [symbol]
      });
    }
    
    // Note: Redis unsubscribe would be handled here in production
    // For mock implementation, subscription cleanup is handled automatically
  }

  // Real-time data fetching with fallback to external APIs
  async fetchRealTimePrice(symbol: string): Promise<PriceData | null> {
    try {
      // First try to get from cache
      const cachedData = await this.getCurrentPrice(symbol);
      if (cachedData && (Date.now() - cachedData.timestamp) < 30000) { // 30 seconds fresh
        return cachedData;
      }

      // Fetch from external API as fallback
      const response = await fetch(`/api/v2/market-data/price/${symbol}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        const priceData: PriceData = {
          symbol: data.symbol,
          price: data.price,
          change: data.change,
          changePercent: data.change_percent,
          volume: data.volume,
          high24h: data.high_24h,
          low24h: data.low_24h,
          marketCap: data.market_cap,
          timestamp: Date.now()
        };
        
        // Update cache
        await this.updatePrice(symbol, priceData);
        return priceData;
      }
    } catch (error) {
      console.error(`Error fetching real-time price for ${symbol}:`, error);
    }

    return null;
  }

  async fetchMultipleRealTimePrices(symbols: string[]): Promise<Record<string, PriceData>> {
    try {
      const response = await fetch('/api/v2/market-data/prices', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbols })
      });

      if (response.ok) {
        const data = await response.json();
        const prices: Record<string, PriceData> = {};
        
        Object.entries(data.prices || {}).forEach(([symbol, priceInfo]: [string, any]) => {
          prices[symbol] = {
            symbol,
            price: priceInfo.price,
            change: priceInfo.change,
            changePercent: priceInfo.change_percent,
            volume: priceInfo.volume,
            high24h: priceInfo.high_24h,
            low24h: priceInfo.low_24h,
            marketCap: priceInfo.market_cap,
            timestamp: Date.now()
          };
          
          // Update individual caches
          this.updatePrice(symbol, prices[symbol]);
        });
        
        return prices;
      }
    } catch (error) {
      console.error('Error fetching multiple real-time prices:', error);
    }

    // Fallback to individual fetches
    return await this.getMultiplePrices(symbols);
  }

  getSubscriptions(): string[] {
    return Array.from(this.subscriptions);
  }

  // Connection Management
  private scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      setTimeout(() => {
        this.reconnectAttempts++;
        this.initializeWebSocket();
      }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
    }
  }

  disconnect(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    this.subscriptions.clear();
    this.emit('disconnected');
  }

  isConnected(): boolean {
    return this.wsConnection?.readyState === WebSocket.OPEN;
  }
}

// Export singleton instance
export const marketDataService = new MarketDataService();
export default MarketDataService;