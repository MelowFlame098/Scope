import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import { redisService } from './RedisService';

// Types for social trading
export interface TraderProfile {
  id: string;
  username: string;
  displayName: string;
  avatar?: string;
  verified: boolean;
  tier: 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond';
  followers: number;
  following: number;
  copiers: number;
  totalCopiedAmount: number;
  joinedAt: Date;
  lastActive: Date;
  bio?: string;
  location?: string;
  tradingExperience: number; // years
  specialties: string[];
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  tradingStyle: 'day_trading' | 'swing_trading' | 'position_trading' | 'scalping';
  preferredAssets: string[];
  socialLinks?: {
    twitter?: string;
    linkedin?: string;
    website?: string;
  };
}

export interface TraderPerformance {
  traderId: string;
  period: '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | 'all';
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  volatility: number;
  winRate: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageHoldingPeriod: number; // hours
  bestTrade: number;
  worstTrade: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  monthlyReturns: { month: string; return: number }[];
  equityCurve: { date: Date; value: number }[];
  drawdownCurve: { date: Date; drawdown: number }[];
  assetAllocation: { asset: string; percentage: number }[];
  tradingFrequency: { period: string; trades: number }[];
}

export interface TradingStrategy {
  id: string;
  name: string;
  description: string;
  creatorId: string;
  creator: TraderProfile;
  category: 'momentum' | 'mean_reversion' | 'breakout' | 'arbitrage' | 'fundamental' | 'technical' | 'quantitative';
  riskLevel: 'low' | 'medium' | 'high';
  minInvestment: number;
  maxInvestment: number;
  currency: string;
  assets: string[];
  timeframe: string;
  isPublic: boolean;
  isActive: boolean;
  subscribers: number;
  totalAUM: number; // Assets Under Management
  performanceFee: number; // percentage
  managementFee: number; // percentage
  createdAt: Date;
  updatedAt: Date;
  performance: TraderPerformance;
  tags: string[];
  methodology: string;
  riskManagement: string;
  backtestResults?: {
    period: string;
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
  };
  reviews: StrategyReview[];
  rating: number;
}

export interface StrategyReview {
  id: string;
  strategyId: string;
  reviewerId: string;
  reviewer: TraderProfile;
  rating: number;
  title: string;
  content: string;
  pros: string[];
  cons: string[];
  wouldRecommend: boolean;
  followedDuration: number; // days
  createdAt: Date;
  helpful: number;
  replies: StrategyReviewReply[];
}

export interface StrategyReviewReply {
  id: string;
  reviewId: string;
  authorId: string;
  author: TraderProfile;
  content: string;
  createdAt: Date;
}

export interface CopyTradingSettings {
  id: string;
  followerId: string;
  traderId: string;
  strategyId?: string;
  isActive: boolean;
  copyMode: 'percentage' | 'fixed_amount' | 'proportional';
  copyPercentage?: number; // for percentage mode
  fixedAmount?: number; // for fixed amount mode
  maxPositionSize: number;
  maxDailyLoss: number;
  maxTotalLoss: number;
  stopLossPercentage?: number;
  takeProfitPercentage?: number;
  copyOnlyProfitableTrades: boolean;
  excludeAssets: string[];
  includeOnlyAssets: string[];
  minTradeAmount: number;
  maxTradeAmount: number;
  delaySeconds: number; // execution delay
  riskMultiplier: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface CopyTrade {
  id: string;
  originalTradeId: string;
  copySettingsId: string;
  followerId: string;
  traderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  originalQuantity: number;
  originalPrice: number;
  executedAt: Date;
  status: 'pending' | 'executed' | 'failed' | 'cancelled';
  pnl?: number;
  fees: number;
  slippage: number;
  delay: number; // milliseconds
  failureReason?: string;
}

export interface SocialPost {
  id: string;
  authorId: string;
  author: TraderProfile;
  type: 'text' | 'trade' | 'analysis' | 'prediction' | 'strategy' | 'poll';
  content: string;
  attachments?: {
    type: 'image' | 'chart' | 'document';
    url: string;
    caption?: string;
  }[];
  tradeData?: {
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
    reasoning: string;
  };
  predictionData?: {
    symbol: string;
    direction: 'up' | 'down';
    targetPrice: number;
    timeframe: string;
    confidence: number;
  };
  pollData?: {
    question: string;
    options: string[];
    votes: { option: string; count: number }[];
    endsAt: Date;
  };
  likes: number;
  comments: number;
  shares: number;
  views: number;
  isPublic: boolean;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface SocialComment {
  id: string;
  postId: string;
  authorId: string;
  author: TraderProfile;
  content: string;
  parentCommentId?: string;
  likes: number;
  replies: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface Leaderboard {
  period: '1d' | '1w' | '1m' | '3m' | '6m' | '1y';
  category: 'return' | 'sharpe' | 'consistency' | 'followers' | 'aum';
  traders: {
    rank: number;
    trader: TraderProfile;
    value: number;
    change: number;
    performance: TraderPerformance;
  }[];
  updatedAt: Date;
}

export interface SocialSignal {
  symbol: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  sources: {
    posts: number;
    predictions: number;
    trades: number;
    mentions: number;
  };
  topTraders: {
    trader: TraderProfile;
    sentiment: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
  }[];
  trendingScore: number;
  momentum: number;
  updatedAt: Date;
}

class SocialTradingService {
  private baseURL: string;
  private socket: Socket | null = null;
  private apiKey: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_SOCIAL_API_URL || 'http://localhost:8002';
    this.apiKey = process.env.REACT_APP_API_KEY || '';
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    this.socket = io(`${this.baseURL}/social`, {
      auth: {
        token: this.apiKey,
      },
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('Connected to social trading service');
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from social trading service');
    });

    this.socket.on('new_trade', (data: CopyTrade) => {
      this.handleNewTrade(data);
    });

    this.socket.on('new_post', (data: SocialPost) => {
      this.handleNewPost(data);
    });

    this.socket.on('trader_update', (data: any) => {
      this.handleTraderUpdate(data);
    });
  }

  // Trader Profiles
  async getTraderProfile(traderId: string): Promise<TraderProfile> {
    try {
      const response = await axios.get(`${this.baseURL}/api/traders/${traderId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get trader profile:', error);
      throw error;
    }
  }

  async updateTraderProfile(updates: Partial<TraderProfile>): Promise<TraderProfile> {
    try {
      const response = await axios.put(`${this.baseURL}/api/traders/profile`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update trader profile:', error);
      throw error;
    }
  }

  async getTraderPerformance(traderId: string, period: string = '1y'): Promise<TraderPerformance> {
    try {
      const response = await axios.get(`${this.baseURL}/api/traders/${traderId}/performance`, {
        params: { period },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get trader performance:', error);
      throw error;
    }
  }

  async searchTraders(filters: any): Promise<TraderProfile[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/traders/search`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to search traders:', error);
      throw error;
    }
  }

  // Following/Followers
  async followTrader(traderId: string): Promise<void> {
    try {
      await axios.post(`${this.baseURL}/api/traders/${traderId}/follow`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
    } catch (error) {
      console.error('Failed to follow trader:', error);
      throw error;
    }
  }

  async unfollowTrader(traderId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseURL}/api/traders/${traderId}/follow`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
    } catch (error) {
      console.error('Failed to unfollow trader:', error);
      throw error;
    }
  }

  async getFollowers(traderId: string): Promise<TraderProfile[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/traders/${traderId}/followers`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get followers:', error);
      throw error;
    }
  }

  async getFollowing(traderId: string): Promise<TraderProfile[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/traders/${traderId}/following`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get following:', error);
      throw error;
    }
  }

  // Copy Trading
  async createCopySettings(settings: Omit<CopyTradingSettings, 'id' | 'createdAt' | 'updatedAt'>): Promise<CopyTradingSettings> {
    try {
      const response = await axios.post(`${this.baseURL}/api/copy-trading/settings`, settings, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create copy settings:', error);
      throw error;
    }
  }

  async updateCopySettings(settingsId: string, updates: Partial<CopyTradingSettings>): Promise<CopyTradingSettings> {
    try {
      const response = await axios.put(`${this.baseURL}/api/copy-trading/settings/${settingsId}`, updates, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to update copy settings:', error);
      throw error;
    }
  }

  async getCopySettings(): Promise<CopyTradingSettings[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/copy-trading/settings`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get copy settings:', error);
      throw error;
    }
  }

  async deleteCopySettings(settingsId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseURL}/api/copy-trading/settings/${settingsId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
    } catch (error) {
      console.error('Failed to delete copy settings:', error);
      throw error;
    }
  }

  async getCopyTrades(filters?: any): Promise<CopyTrade[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/copy-trading/trades`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get copy trades:', error);
      throw error;
    }
  }

  // Strategy Marketplace
  async getStrategies(filters?: any): Promise<TradingStrategy[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/strategies`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get strategies:', error);
      throw error;
    }
  }

  async getStrategy(strategyId: string): Promise<TradingStrategy> {
    try {
      const response = await axios.get(`${this.baseURL}/api/strategies/${strategyId}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get strategy:', error);
      throw error;
    }
  }

  async createStrategy(strategy: Omit<TradingStrategy, 'id' | 'createdAt' | 'updatedAt' | 'creator' | 'performance' | 'reviews' | 'rating'>): Promise<TradingStrategy> {
    try {
      const response = await axios.post(`${this.baseURL}/api/strategies`, strategy, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create strategy:', error);
      throw error;
    }
  }

  async subscribeToStrategy(strategyId: string, settings: any): Promise<void> {
    try {
      await axios.post(`${this.baseURL}/api/strategies/${strategyId}/subscribe`, settings, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
    } catch (error) {
      console.error('Failed to subscribe to strategy:', error);
      throw error;
    }
  }

  async unsubscribeFromStrategy(strategyId: string): Promise<void> {
    try {
      await axios.delete(`${this.baseURL}/api/strategies/${strategyId}/subscribe`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
    } catch (error) {
      console.error('Failed to unsubscribe from strategy:', error);
      throw error;
    }
  }

  // Strategy Reviews
  async createStrategyReview(review: Omit<StrategyReview, 'id' | 'reviewer' | 'createdAt' | 'helpful' | 'replies'>): Promise<StrategyReview> {
    try {
      const response = await axios.post(`${this.baseURL}/api/strategies/${review.strategyId}/reviews`, review, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create strategy review:', error);
      throw error;
    }
  }

  async getStrategyReviews(strategyId: string): Promise<StrategyReview[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/strategies/${strategyId}/reviews`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get strategy reviews:', error);
      throw error;
    }
  }

  // Social Feed
  async createPost(post: Omit<SocialPost, 'id' | 'author' | 'likes' | 'comments' | 'shares' | 'views' | 'createdAt' | 'updatedAt'>): Promise<SocialPost> {
    try {
      const response = await axios.post(`${this.baseURL}/api/social/posts`, post, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create post:', error);
      throw error;
    }
  }

  async getFeed(filters?: any): Promise<SocialPost[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/social/feed`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get feed:', error);
      throw error;
    }
  }

  async likePost(postId: string): Promise<void> {
    try {
      await axios.post(`${this.baseURL}/api/social/posts/${postId}/like`, {}, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
    } catch (error) {
      console.error('Failed to like post:', error);
      throw error;
    }
  }

  async commentOnPost(postId: string, content: string, parentCommentId?: string): Promise<SocialComment> {
    try {
      const response = await axios.post(`${this.baseURL}/api/social/posts/${postId}/comments`, {
        content,
        parentCommentId,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to comment on post:', error);
      throw error;
    }
  }

  // Leaderboards with Redis integration
  async getLeaderboard(period: string = '1m', category: string = 'return'): Promise<Leaderboard> {
    try {
      // Try to get from Redis first for real-time data
      const redisLeaderboard = await this.getRedisLeaderboard(category, period, 50);
      
      if (redisLeaderboard.length > 0) {
        // Convert Redis data to Leaderboard format
        const traders = await Promise.all(
          redisLeaderboard.map(async (entry, index) => {
            try {
              const traderProfile = await this.getTraderProfile(entry.userId);
              const performance = await this.getTraderPerformance(entry.userId, period);
              
              return {
                rank: entry.rank,
                trader: traderProfile,
                value: entry.score,
                change: 0, // TODO: Calculate change from previous period
                performance
              };
            } catch (error) {
              console.warn(`Failed to get trader data for ${entry.userId}:`, error);
              return null;
            }
          })
        );

        return {
          period: period as any,
          category: category as any,
          traders: traders.filter(t => t !== null) as any[],
          updatedAt: new Date()
        };
      }

      // Fallback to API if Redis is empty
      const response = await axios.get(`${this.baseURL}/api/leaderboard`, {
        params: { period, category },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get leaderboard:', error);
      throw error;
    }
  }

  // Redis-based leaderboard methods
  private async getRedisLeaderboard(category: string, period?: string, limit: number = 10) {
    try {
      if (period && ['1d', '1w', '1m'].includes(period)) {
        const periodMap = { '1d': 'daily', '1w': 'weekly', '1m': 'monthly' };
        return await redisService.getPeriodLeaderboard(category, periodMap[period as keyof typeof periodMap] as any, limit);
      }
      return await redisService.getTopTraders(category, limit);
    } catch (error) {
      console.error('Failed to get Redis leaderboard:', error);
      return [];
    }
  }

  async updateTraderMetrics(traderId: string, metrics: {
    totalReturn: number;
    winRate: number;
    sharpeRatio: number;
    totalTrades: number;
    followers: number;
    aum: number;
  }): Promise<void> {
    try {
      // Update Redis leaderboards
      await redisService.updateTradingPerformance(traderId, metrics);
      
      // Update period-based leaderboards
      await Promise.all([
        redisService.updatePeriodLeaderboard(traderId, metrics.totalReturn, 'return', 'daily'),
        redisService.updatePeriodLeaderboard(traderId, metrics.totalReturn, 'return', 'weekly'),
        redisService.updatePeriodLeaderboard(traderId, metrics.totalReturn, 'return', 'monthly'),
        redisService.updatePeriodLeaderboard(traderId, metrics.winRate, 'winrate', 'daily'),
        redisService.updatePeriodLeaderboard(traderId, metrics.sharpeRatio, 'sharpe', 'daily'),
        redisService.updatePeriodLeaderboard(traderId, metrics.followers, 'followers', 'daily'),
        redisService.updatePeriodLeaderboard(traderId, metrics.aum, 'aum', 'daily')
      ]);

      console.log(`Updated leaderboard metrics for trader ${traderId}`);
    } catch (error) {
      console.error('Failed to update trader metrics:', error);
    }
  }

  async getUserRank(userId: string, category: string = 'return'): Promise<number | null> {
    try {
      return await redisService.getUserRank(userId, category);
    } catch (error) {
      console.error('Failed to get user rank:', error);
      return null;
    }
  }

  // Enhanced leaderboard categories
  async getMultiCategoryLeaderboard(categories: string[] = ['return', 'sharpe', 'winrate', 'followers'], limit: number = 10): Promise<{
    [category: string]: Array<{
      userId: string;
      score: number;
      rank: number;
      metrics?: any;
    }>;
  }> {
    try {
      const results: any = {};
      
      await Promise.all(
        categories.map(async (category) => {
          results[category] = await redisService.getTopTraders(category, limit);
        })
      );
      
      return results;
    } catch (error) {
      console.error('Failed to get multi-category leaderboard:', error);
      return {};
    }
  }

  // Social Signals
  async getSocialSignals(symbols?: string[]): Promise<SocialSignal[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/social/signals`, {
        params: { symbols: symbols?.join(',') },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get social signals:', error);
      throw error;
    }
  }

  async getSocialSignal(symbol: string): Promise<SocialSignal> {
    try {
      const response = await axios.get(`${this.baseURL}/api/social/signals/${symbol}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get social signal:', error);
      throw error;
    }
  }

  // Real-time subscriptions
  subscribeToTrader(traderId: string, callback: (data: any) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_trader', { traderId });
      this.socket.on('trader_update', callback);
    }
  }

  subscribeToFeed(callback: (post: SocialPost) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_feed');
      this.socket.on('new_post', callback);
    }
  }

  subscribeToCopyTrades(callback: (trade: CopyTrade) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_copy_trades');
      this.socket.on('new_trade', callback);
    }
  }

  // Event handlers
  private handleNewTrade(trade: CopyTrade) {
    console.log('New copy trade:', trade);
  }

  private handleNewPost(post: SocialPost) {
    console.log('New social post:', post);
  }

  private handleTraderUpdate(data: any) {
    console.log('Trader update:', data);
  }

  // Cleanup
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const socialTradingService = new SocialTradingService();
export default socialTradingService;