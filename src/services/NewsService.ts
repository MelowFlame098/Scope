import { EventEmitter } from 'events';
import { redisService } from './RedisService';

export interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  content: string;
  source: string;
  author: string;
  publishedAt: string;
  url: string;
  imageUrl?: string;
  symbols: string[];
  sentiment: 'bullish' | 'bearish' | 'neutral';
  sentimentScore: number; // -1 to 1
  impact: 'high' | 'medium' | 'low';
  category: 'earnings' | 'merger' | 'regulatory' | 'market' | 'crypto' | 'general';
  tags: string[];
  readTime: number; // in minutes
}

export interface MarketNews {
  breakingNews: NewsArticle[];
  topStories: NewsArticle[];
  assetSpecific: NewsArticle[];
  earnings: NewsArticle[];
  regulatory: NewsArticle[];
  crypto: NewsArticle[];
}

export interface NewsFilter {
  symbols?: string[];
  categories?: string[];
  sentiment?: ('bullish' | 'bearish' | 'neutral')[];
  impact?: ('high' | 'medium' | 'low')[];
  sources?: string[];
  timeRange?: '1h' | '6h' | '24h' | '7d' | '30d';
}

class NewsService extends EventEmitter {
  private newsCache: Map<string, NewsArticle[]> = new Map();
  private wsConnection: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor() {
    super();
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    try {
      // In production, this would connect to your news WebSocket endpoint
      // For now, we'll simulate real-time updates
      this.simulateRealTimeUpdates();
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  private simulateRealTimeUpdates() {
    // Simulate real-time news updates every 30 seconds
    setInterval(() => {
      const newArticle = this.generateMockNewsArticle();
      this.emit('newArticle', newArticle);
      
      // Update cache
      const symbol = newArticle.symbols[0] || 'GENERAL';
      const existing = this.newsCache.get(symbol) || [];
      this.newsCache.set(symbol, [newArticle, ...existing.slice(0, 49)]); // Keep latest 50
    }, 30000);

    // Simulate breaking news occasionally
    setInterval(() => {
      if (Math.random() < 0.1) { // 10% chance every minute
        const breakingNews = this.generateBreakingNews();
        this.emit('breakingNews', breakingNews);
      }
    }, 60000);
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      setTimeout(() => {
        this.reconnectAttempts++;
        this.initializeWebSocket();
      }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
    }
  }

  private generateMockNewsArticle(): NewsArticle {
    const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'ETH-USD', 'SPY'];
    const sources = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance', 'Financial Times', 'WSJ'];
    const categories: NewsArticle['category'][] = ['earnings', 'merger', 'regulatory', 'market', 'crypto', 'general'];
    const sentiments: NewsArticle['sentiment'][] = ['bullish', 'bearish', 'neutral'];
    const impacts: NewsArticle['impact'][] = ['high', 'medium', 'low'];

    const symbol = symbols[Math.floor(Math.random() * symbols.length)];
    const source = sources[Math.floor(Math.random() * sources.length)];
    const category = categories[Math.floor(Math.random() * categories.length)];
    const sentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
    const impact = impacts[Math.floor(Math.random() * impacts.length)];

    const titles = {
      earnings: [
        `${symbol} Reports Q4 Earnings Beat Expectations`,
        `${symbol} Quarterly Revenue Surges 15% Year-over-Year`,
        `${symbol} Misses Earnings Estimates, Stock Drops in After-Hours`
      ],
      merger: [
        `${symbol} Announces Strategic Acquisition Deal`,
        `Merger Talks Between ${symbol} and Industry Leader Heat Up`,
        `${symbol} Completes $2B Acquisition, Expands Market Reach`
      ],
      regulatory: [
        `SEC Announces New Regulations Affecting ${symbol}`,
        `${symbol} Faces Regulatory Scrutiny Over Data Practices`,
        `Government Approves ${symbol}'s Compliance Framework`
      ],
      market: [
        `${symbol} Hits New All-Time High Amid Market Rally`,
        `Analysts Upgrade ${symbol} Price Target to $200`,
        `${symbol} Shows Strong Technical Breakout Pattern`
      ],
      crypto: [
        `${symbol} Surges 20% Following Institutional Adoption`,
        `Major Exchange Lists ${symbol}, Trading Volume Spikes`,
        `${symbol} Network Upgrade Promises Faster Transactions`
      ],
      general: [
        `${symbol} Announces New Product Launch Strategy`,
        `${symbol} CEO Discusses Future Growth Plans`,
        `${symbol} Partners with Tech Giant for Innovation`
      ]
    };

    const categoryTitles = titles[category];
    const title = categoryTitles[Math.floor(Math.random() * categoryTitles.length)];

    return {
      id: `news_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title,
      summary: `Latest developments regarding ${symbol} show ${sentiment} sentiment with ${impact} market impact. ${source} reports significant movement in the ${category} sector.`,
      content: `Full article content would be here. This is a detailed analysis of the ${symbol} situation, providing comprehensive coverage of the ${category} developments. The market has shown ${sentiment} reaction to these developments, with analysts predicting ${impact} impact on the stock price.`,
      source,
      author: `${source} Editorial Team`,
      publishedAt: new Date().toISOString(),
      url: `https://${source.toLowerCase().replace(' ', '')}.com/news/${symbol.toLowerCase()}-${Date.now()}`,
      imageUrl: `https://picsum.photos/400/200?random=${Math.floor(Math.random() * 1000)}`,
      symbols: [symbol],
      sentiment,
      sentimentScore: sentiment === 'bullish' ? 0.3 + Math.random() * 0.7 : 
                     sentiment === 'bearish' ? -0.3 - Math.random() * 0.7 : 
                     (Math.random() - 0.5) * 0.6,
      impact,
      category,
      tags: [symbol, category, sentiment, source.toLowerCase()],
      readTime: Math.floor(Math.random() * 5) + 1
    };
  }

  private generateBreakingNews(): NewsArticle {
    const breakingTitles = [
      'BREAKING: Federal Reserve Announces Emergency Rate Decision',
      'BREAKING: Major Tech Stock Halted Due to Unusual Activity',
      'BREAKING: Cryptocurrency Market Sees Massive Institutional Inflow',
      'BREAKING: Geopolitical Tensions Impact Global Markets',
      'BREAKING: Major Earnings Surprise Shakes Market Sentiment'
    ];

    const article = this.generateMockNewsArticle();
    return {
      ...article,
      title: breakingTitles[Math.floor(Math.random() * breakingTitles.length)],
      impact: 'high',
      category: 'market'
    };
  }

  async getNews(filter: NewsFilter = {}): Promise<MarketNews> {
    // Try to get cached news first
    const cacheKey = `news:filtered:${JSON.stringify(filter)}`;
    const cachedNews = await redisService.get<MarketNews>(cacheKey);
    
    if (cachedNews) {
      return cachedNews;
    }

    // Generate fresh news data
    const newsData = this.generateMockNewsData(filter);
    
    // Cache the news data for 5 minutes
    await redisService.set(cacheKey, newsData, 300);
    
    return newsData;
  }

  async getAssetNews(symbol: string, limit: number = 20): Promise<NewsArticle[]> {
    // Try to get cached asset news first
    const cacheKey = `news:asset:${symbol}:${limit}`;
    const cachedNews = await redisService.get<NewsArticle[]>(cacheKey);
    
    if (cachedNews) {
      return cachedNews;
    }

    // Generate fresh asset-specific news
    const allNews = this.generateMockNewsData({ symbols: [symbol] });
    const assetNews = [
      ...allNews.assetSpecific,
      ...allNews.breakingNews.filter(article => article.symbols.includes(symbol)),
      ...allNews.topStories.filter(article => article.symbols.includes(symbol))
    ].slice(0, limit);

    // Cache asset news for 3 minutes
    await redisService.set(cacheKey, assetNews, 180);
    
    return assetNews;
  }

  private generateMockNewsData(filter: NewsFilter): MarketNews {
    const generateArticles = (count: number, categoryFilter?: string): NewsArticle[] => {
      const articles: NewsArticle[] = [];
      for (let i = 0; i < count; i++) {
        const article = this.generateMockNewsArticle();
        if (categoryFilter) {
          article.category = categoryFilter as NewsArticle['category'];
        }
        articles.push(article);
      }
      return articles;
    };

    return {
      breakingNews: generateArticles(3).map(article => ({ ...article, impact: 'high' })),
      topStories: generateArticles(10),
      assetSpecific: filter.symbols ? 
        filter.symbols.flatMap(symbol => 
          generateArticles(5).map(article => ({ ...article, symbols: [symbol] }))
        ) : generateArticles(15),
      earnings: generateArticles(8, 'earnings'),
      regulatory: generateArticles(6, 'regulatory'),
      crypto: generateArticles(10, 'crypto')
    };
  }

  async searchNews(query: string, limit: number = 20): Promise<NewsArticle[]> {
    // Try to get cached search results first
    const cacheKey = `news:search:${query.toLowerCase()}:${limit}`;
    const cachedResults = await redisService.get<NewsArticle[]>(cacheKey);
    
    if (cachedResults) {
      return cachedResults;
    }

    // Generate fresh search results
    const allNews = this.generateMockNewsData({});
    const allArticles = [
      ...allNews.breakingNews,
      ...allNews.topStories,
      ...allNews.assetSpecific,
      ...allNews.earnings,
      ...allNews.regulatory,
      ...allNews.crypto
    ];

    const searchResults = allArticles.filter(article =>
      article.title.toLowerCase().includes(query.toLowerCase()) ||
      article.summary.toLowerCase().includes(query.toLowerCase()) ||
      article.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase())) ||
      article.symbols.some(symbol => symbol.toLowerCase().includes(query.toLowerCase()))
    ).slice(0, limit);

    // Cache search results for 2 minutes
    await redisService.set(cacheKey, searchResults, 120);
    
    return searchResults;
  }

  subscribeToSymbol(symbol: string) {
    // In production, this would subscribe to real-time updates for the symbol
    console.log(`Subscribed to news updates for ${symbol}`);
  }

  unsubscribeFromSymbol(symbol: string) {
    // In production, this would unsubscribe from real-time updates
    console.log(`Unsubscribed from news updates for ${symbol}`);
  }

  getSentimentAnalysis(articles: NewsArticle[]): {
    overall: 'bullish' | 'bearish' | 'neutral';
    score: number;
    distribution: { bullish: number; bearish: number; neutral: number };
  } {
    if (articles.length === 0) {
      return {
        overall: 'neutral',
        score: 0,
        distribution: { bullish: 0, bearish: 0, neutral: 0 }
      };
    }

    const distribution = articles.reduce((acc, article) => {
      acc[article.sentiment]++;
      return acc;
    }, { bullish: 0, bearish: 0, neutral: 0 });

    const totalScore = articles.reduce((sum, article) => sum + article.sentimentScore, 0);
    const averageScore = totalScore / articles.length;

    const overall: 'bullish' | 'bearish' | 'neutral' = 
      averageScore > 0.1 ? 'bullish' : 
      averageScore < -0.1 ? 'bearish' : 'neutral';

    return {
      overall,
      score: averageScore,
      distribution
    };
  }

  disconnect() {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
    this.removeAllListeners();
  }
}

export const newsService = new NewsService();
export default NewsService;