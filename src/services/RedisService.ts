// Simple EventEmitter replacement for Edge Runtime compatibility
class SimpleEventEmitter {
  private events: Map<string, Function[]> = new Map();

  on(event: string, callback: Function): void {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(callback);
  }

  emit(event: string, ...args: any[]): void {
    const callbacks = this.events.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(...args));
    }
  }

  removeListener(event: string, callback: Function): void {
    const callbacks = this.events.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }
}

// Redis Client Interface for type safety
interface RedisClient {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, options?: { EX?: number; PX?: number }): Promise<string>;
  del(key: string): Promise<number>;
  exists(key: string): Promise<number>;
  expire(key: string, seconds: number): Promise<number>;
  hget(key: string, field: string): Promise<string | null>;
  hset(key: string, field: string, value: string): Promise<number>;
  hgetall(key: string): Promise<Record<string, string>>;
  hdel(key: string, field: string): Promise<number>;
  lpush(key: string, value: string): Promise<number>;
  rpush(key: string, value: string): Promise<number>;
  lpop(key: string): Promise<string | null>;
  rpop(key: string): Promise<string | null>;
  lrange(key: string, start: number, stop: number): Promise<string[]>;
  zadd(key: string, score: number, member: string): Promise<number>;
  zrange(key: string, start: number, stop: number, options?: { WITHSCORES?: boolean }): Promise<string[]>;
  zrevrange(key: string, start: number, stop: number, options?: { WITHSCORES?: boolean }): Promise<string[]>;
  zrem(key: string, member: string): Promise<number>;
  publish(channel: string, message: string): Promise<number>;
  subscribe(channel: string): Promise<void>;
  unsubscribe(channel: string): Promise<void>;
  on(event: string, callback: Function): void;
  disconnect(): Promise<void>;
}

// Mock Redis Client for development (when Redis is not available)
class MockRedisClient extends SimpleEventEmitter implements RedisClient {
  private data: Map<string, any> = new Map();
  private hashes: Map<string, Map<string, string>> = new Map();
  private lists: Map<string, string[]> = new Map();
  private sortedSets: Map<string, Map<string, number>> = new Map();
  private subscribers: Map<string, Set<Function>> = new Map();

  async get(key: string): Promise<string | null> {
    return this.data.get(key) || null;
  }

  async set(key: string, value: string, options?: { EX?: number; PX?: number }): Promise<string> {
    this.data.set(key, value);
    if (options?.EX) {
      setTimeout(() => this.data.delete(key), options.EX * 1000);
    }
    if (options?.PX) {
      setTimeout(() => this.data.delete(key), options.PX);
    }
    return 'OK';
  }

  async del(key: string): Promise<number> {
    const existed = this.data.has(key);
    this.data.delete(key);
    this.hashes.delete(key);
    this.lists.delete(key);
    this.sortedSets.delete(key);
    return existed ? 1 : 0;
  }

  async exists(key: string): Promise<number> {
    return this.data.has(key) ? 1 : 0;
  }

  async expire(key: string, seconds: number): Promise<number> {
    if (this.data.has(key)) {
      setTimeout(() => this.data.delete(key), seconds * 1000);
      return 1;
    }
    return 0;
  }

  async hget(key: string, field: string): Promise<string | null> {
    const hash = this.hashes.get(key);
    return hash?.get(field) || null;
  }

  async hset(key: string, field: string, value: string): Promise<number> {
    if (!this.hashes.has(key)) {
      this.hashes.set(key, new Map());
    }
    const hash = this.hashes.get(key)!;
    const isNew = !hash.has(field);
    hash.set(field, value);
    return isNew ? 1 : 0;
  }

  async hgetall(key: string): Promise<Record<string, string>> {
    const hash = this.hashes.get(key);
    if (!hash) return {};
    return Object.fromEntries(hash.entries());
  }

  async hdel(key: string, field: string): Promise<number> {
    const hash = this.hashes.get(key);
    if (hash && hash.has(field)) {
      hash.delete(field);
      return 1;
    }
    return 0;
  }

  async lpush(key: string, value: string): Promise<number> {
    if (!this.lists.has(key)) {
      this.lists.set(key, []);
    }
    const list = this.lists.get(key)!;
    list.unshift(value);
    return list.length;
  }

  async rpush(key: string, value: string): Promise<number> {
    if (!this.lists.has(key)) {
      this.lists.set(key, []);
    }
    const list = this.lists.get(key)!;
    list.push(value);
    return list.length;
  }

  async lpop(key: string): Promise<string | null> {
    const list = this.lists.get(key);
    return list?.shift() || null;
  }

  async rpop(key: string): Promise<string | null> {
    const list = this.lists.get(key);
    return list?.pop() || null;
  }

  async lrange(key: string, start: number, stop: number): Promise<string[]> {
    const list = this.lists.get(key) || [];
    return list.slice(start, stop === -1 ? undefined : stop + 1);
  }

  async zadd(key: string, score: number, member: string): Promise<number> {
    if (!this.sortedSets.has(key)) {
      this.sortedSets.set(key, new Map());
    }
    const sortedSet = this.sortedSets.get(key)!;
    const isNew = !sortedSet.has(member);
    sortedSet.set(member, score);
    return isNew ? 1 : 0;
  }

  async zrange(key: string, start: number, stop: number, options?: { WITHSCORES?: boolean }): Promise<string[]> {
    const sortedSet = this.sortedSets.get(key);
    if (!sortedSet) return [];
    
    const sorted = Array.from(sortedSet.entries()).sort((a, b) => a[1] - b[1]);
    const slice = sorted.slice(start, stop === -1 ? undefined : stop + 1);
    
    if (options?.WITHSCORES) {
      return slice.flatMap(([member, score]) => [member, score.toString()]);
    }
    return slice.map(([member]) => member);
  }

  async zrevrange(key: string, start: number, stop: number, options?: { WITHSCORES?: boolean }): Promise<string[]> {
    const sortedSet = this.sortedSets.get(key);
    if (!sortedSet) return [];
    
    const sorted = Array.from(sortedSet.entries()).sort((a, b) => b[1] - a[1]);
    const slice = sorted.slice(start, stop === -1 ? undefined : stop + 1);
    
    if (options?.WITHSCORES) {
      return slice.flatMap(([member, score]) => [member, score.toString()]);
    }
    return slice.map(([member]) => member);
  }

  async zrem(key: string, member: string): Promise<number> {
    const sortedSet = this.sortedSets.get(key);
    if (sortedSet && sortedSet.has(member)) {
      sortedSet.delete(member);
      return 1;
    }
    return 0;
  }

  async publish(channel: string, message: string): Promise<number> {
    const subscribers = this.subscribers.get(channel);
    if (subscribers) {
      subscribers.forEach(callback => callback(message));
      return subscribers.size;
    }
    return 0;
  }

  async subscribe(channel: string): Promise<void> {
    // Mock subscription - in real implementation, this would set up Redis subscription
    console.log(`Subscribed to channel: ${channel}`);
  }

  async unsubscribe(channel: string): Promise<void> {
    this.subscribers.delete(channel);
    console.log(`Unsubscribed from channel: ${channel}`);
  }

  async disconnect(): Promise<void> {
    this.data.clear();
    this.hashes.clear();
    this.lists.clear();
    this.sortedSets.clear();
    this.subscribers.clear();
  }
}

// Cache Configuration
interface CacheConfig {
  defaultTTL: number; // Time to live in seconds
  maxRetries: number;
  retryDelay: number;
}

// Redis Service Class
class RedisService extends SimpleEventEmitter {
  private client: RedisClient;
  private isConnected: boolean = false;
  private config: CacheConfig;

  constructor(config: Partial<CacheConfig> = {}) {
    super();
    this.config = {
      defaultTTL: 3600, // 1 hour
      maxRetries: 3,
      retryDelay: 1000,
      ...config
    };

    // Initialize with mock client for development
    this.client = new MockRedisClient();
    this.isConnected = true;
    
    // In production, you would initialize with actual Redis client:
    // this.initializeRedisClient();
  }

  // Connection Management
  async connect(): Promise<void> {
    try {
      // In production, establish actual Redis connection
      this.isConnected = true;
      this.emit('connected');
      console.log('Redis service connected (mock mode)');
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.disconnect();
      this.isConnected = false;
      this.emit('disconnected');
    }
  }

  // Basic Cache Operations
  async get<T>(key: string): Promise<T | null> {
    try {
      const value = await this.client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error(`Redis GET error for key ${key}:`, error);
      return null;
    }
  }

  async set<T>(key: string, value: T, ttl?: number): Promise<boolean> {
    try {
      const serialized = JSON.stringify(value);
      const options = ttl ? { EX: ttl } : { EX: this.config.defaultTTL };
      await this.client.set(key, serialized, options);
      return true;
    } catch (error) {
      console.error(`Redis SET error for key ${key}:`, error);
      return false;
    }
  }

  async del(key: string): Promise<boolean> {
    try {
      const result = await this.client.del(key);
      return result > 0;
    } catch (error) {
      console.error(`Redis DEL error for key ${key}:`, error);
      return false;
    }
  }

  async exists(key: string): Promise<boolean> {
    try {
      const result = await this.client.exists(key);
      return result > 0;
    } catch (error) {
      console.error(`Redis EXISTS error for key ${key}:`, error);
      return false;
    }
  }

  // Hash Operations (for user sessions, settings)
  async hget<T>(key: string, field: string): Promise<T | null> {
    try {
      const value = await this.client.hget(key, field);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error(`Redis HGET error for key ${key}, field ${field}:`, error);
      return null;
    }
  }

  async hset<T>(key: string, field: string, value: T): Promise<boolean> {
    try {
      const serialized = JSON.stringify(value);
      await this.client.hset(key, field, serialized);
      return true;
    } catch (error) {
      console.error(`Redis HSET error for key ${key}, field ${field}:`, error);
      return false;
    }
  }

  async hgetall<T>(key: string): Promise<Record<string, T>> {
    try {
      const hash = await this.client.hgetall(key);
      const result: Record<string, T> = {};
      for (const [field, value] of Object.entries(hash)) {
        try {
          result[field] = JSON.parse(value);
        } catch {
          result[field] = value as T;
        }
      }
      return result;
    } catch (error) {
      console.error(`Redis HGETALL error for key ${key}:`, error);
      return {};
    }
  }

  async hdel(key: string, field: string): Promise<number> {
    try {
      return await this.client.hdel(key, field);
    } catch (error) {
      console.error(`Redis HDEL error for key ${key}, field ${field}:`, error);
      return 0;
    }
  }

  // List Operations (for queues, recent items)
  async lpush<T>(key: string, value: T): Promise<number> {
    try {
      const serialized = JSON.stringify(value);
      return await this.client.lpush(key, serialized);
    } catch (error) {
      console.error(`Redis LPUSH error for key ${key}:`, error);
      return 0;
    }
  }

  async rpop<T>(key: string): Promise<T | null> {
    try {
      const value = await this.client.rpop(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error(`Redis RPOP error for key ${key}:`, error);
      return null;
    }
  }

  async lrange<T>(key: string, start: number = 0, stop: number = -1): Promise<T[]> {
    try {
      const values = await this.client.lrange(key, start, stop);
      return values.map(value => {
        try {
          return JSON.parse(value);
        } catch {
          return value as T;
        }
      });
    } catch (error) {
      console.error(`Redis LRANGE error for key ${key}:`, error);
      return [];
    }
  }

  // Sorted Set Operations (for leaderboards, rankings)
  async zadd(key: string, score: number, member: string): Promise<boolean> {
    try {
      const result = await this.client.zadd(key, score, member);
      return result > 0;
    } catch (error) {
      console.error(`Redis ZADD error for key ${key}:`, error);
      return false;
    }
  }

  async zrevrange(key: string, start: number = 0, stop: number = -1, withScores: boolean = false): Promise<string[]> {
    try {
      return await this.client.zrevrange(key, start, stop, { WITHSCORES: withScores });
    } catch (error) {
      console.error(`Redis ZREVRANGE error for key ${key}:`, error);
      return [];
    }
  }

  // Pub/Sub Operations
  async publish(channel: string, message: any): Promise<number> {
    try {
      const serialized = typeof message === 'string' ? message : JSON.stringify(message);
      return await this.client.publish(channel, serialized);
    } catch (error) {
      console.error(`Redis PUBLISH error for channel ${channel}:`, error);
      return 0;
    }
  }

  async subscribe(channel: string, callback: (message: any) => void): Promise<void> {
    try {
      await this.client.subscribe(channel);
      this.client.on('message', (receivedChannel: string, message: string) => {
        if (receivedChannel === channel) {
          try {
            const parsed = JSON.parse(message);
            callback(parsed);
          } catch {
            callback(message);
          }
        }
      });
    } catch (error) {
      console.error(`Redis SUBSCRIBE error for channel ${channel}:`, error);
    }
  }

  // High-level Cache Methods
  async cacheMarketData(symbol: string, data: any, ttl: number = 60): Promise<void> {
    await this.set(`market:${symbol}`, data, ttl);
  }

  async getMarketData(symbol: string): Promise<any> {
    return await this.get(`market:${symbol}`);
  }

  async cacheNews(newsId: string, article: any, ttl: number = 1800): Promise<void> {
    await this.set(`news:${newsId}`, article, ttl);
  }

  async getNews(newsId: string): Promise<any> {
    return await this.get(`news:${newsId}`);
  }

  async cacheUserSession(userId: string, sessionData: any, ttl: number = 86400): Promise<void> {
    await this.hset(`session:${userId}`, 'data', sessionData);
    await this.client.expire(`session:${userId}`, ttl);
  }

  async getUserSession(userId: string): Promise<any> {
    return await this.hget(`session:${userId}`, 'data');
  }

  // Trading Queue Operations
  async addTradeOrder(order: any): Promise<void> {
    await this.lpush('trade:orders', order);
  }

  async getNextTradeOrder(): Promise<any> {
    return await this.rpop('trade:orders');
  }

  // Enhanced Leaderboard Operations with multiple categories
  async updateLeaderboard(userId: string, score: number, category: string = 'trading'): Promise<void> {
    await this.zadd(`leaderboard:${category}`, score, userId);
  }

  async getLeaderboard(category: string = 'trading', limit: number = 10): Promise<string[]> {
    return await this.zrevrange(`leaderboard:${category}`, 0, limit - 1, true);
  }

  // Trading Performance Leaderboards
  async updateTradingPerformance(userId: string, metrics: {
    totalReturn: number;
    winRate: number;
    sharpeRatio: number;
    totalTrades: number;
    followers: number;
    aum: number;
  }): Promise<void> {
    const timestamp = Date.now();
    
    // Update multiple leaderboards
    await Promise.all([
      this.zadd('leaderboard:return', metrics.totalReturn, userId),
      this.zadd('leaderboard:winrate', metrics.winRate, userId),
      this.zadd('leaderboard:sharpe', metrics.sharpeRatio, userId),
      this.zadd('leaderboard:trades', metrics.totalTrades, userId),
      this.zadd('leaderboard:followers', metrics.followers, userId),
      this.zadd('leaderboard:aum', metrics.aum, userId),
      
      // Store user metrics with timestamp
      this.hset(`user:${userId}:metrics`, 'totalReturn', metrics.totalReturn.toString()),
      this.hset(`user:${userId}:metrics`, 'winRate', metrics.winRate.toString()),
      this.hset(`user:${userId}:metrics`, 'sharpeRatio', metrics.sharpeRatio.toString()),
      this.hset(`user:${userId}:metrics`, 'totalTrades', metrics.totalTrades.toString()),
      this.hset(`user:${userId}:metrics`, 'followers', metrics.followers.toString()),
      this.hset(`user:${userId}:metrics`, 'aum', metrics.aum.toString()),
      this.hset(`user:${userId}:metrics`, 'lastUpdated', timestamp.toString())
    ]);
  }

  async getTopTraders(category: string = 'return', limit: number = 10): Promise<Array<{
    userId: string;
    score: number;
    rank: number;
    metrics?: any;
  }>> {
    try {
      const results = await this.zrevrange(`leaderboard:${category}`, 0, limit - 1, true);
      const traders = [];
      
      for (let i = 0; i < results.length; i += 2) {
        const userId = results[i];
        const score = parseFloat(results[i + 1]);
        
        // Get additional metrics for the user
        const metrics = await this.hgetall(`user:${userId}:metrics`);
        
        traders.push({
          userId,
          score,
          rank: Math.floor(i / 2) + 1,
          metrics: metrics ? {
            totalReturn: parseFloat(metrics.totalReturn || '0'),
            winRate: parseFloat(metrics.winRate || '0'),
            sharpeRatio: parseFloat(metrics.sharpeRatio || '0'),
            totalTrades: parseInt(metrics.totalTrades || '0'),
            followers: parseInt(metrics.followers || '0'),
            aum: parseFloat(metrics.aum || '0'),
            lastUpdated: parseInt(metrics.lastUpdated || '0')
          } : null
        });
      }
      
      return traders;
    } catch (error) {
      console.error('Error getting top traders:', error);
      return [];
    }
  }

  async getUserRank(userId: string, category: string = 'return'): Promise<number | null> {
    try {
      const rank = await this.client.zrevrank(`leaderboard:${category}`, userId);
      return rank !== null ? rank + 1 : null;
    } catch (error) {
      console.error('Error getting user rank:', error);
      return null;
    }
  }

  // Period-based leaderboards (daily, weekly, monthly)
  async updatePeriodLeaderboard(userId: string, score: number, category: string, period: 'daily' | 'weekly' | 'monthly'): Promise<void> {
    const now = new Date();
    let periodKey = '';
    
    switch (period) {
      case 'daily':
        periodKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        break;
      case 'weekly':
        const weekStart = new Date(now.setDate(now.getDate() - now.getDay()));
        periodKey = `${weekStart.getFullYear()}-W${Math.ceil(weekStart.getDate() / 7)}`;
        break;
      case 'monthly':
        periodKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
        break;
    }
    
    await this.zadd(`leaderboard:${category}:${period}:${periodKey}`, score, userId);
    
    // Set expiration for period-based leaderboards
    const expiration = period === 'daily' ? 86400 * 7 : // Keep daily for 7 days
                      period === 'weekly' ? 86400 * 30 : // Keep weekly for 30 days
                      86400 * 365; // Keep monthly for 1 year
    
    await this.client.expire(`leaderboard:${category}:${period}:${periodKey}`, expiration);
  }

  async getPeriodLeaderboard(category: string, period: 'daily' | 'weekly' | 'monthly', limit: number = 10): Promise<Array<{
    userId: string;
    score: number;
    rank: number;
  }>> {
    const now = new Date();
    let periodKey = '';
    
    switch (period) {
      case 'daily':
        periodKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
        break;
      case 'weekly':
        const weekStart = new Date(now.setDate(now.getDate() - now.getDay()));
        periodKey = `${weekStart.getFullYear()}-W${Math.ceil(weekStart.getDate() / 7)}`;
        break;
      case 'monthly':
        periodKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
        break;
    }
    
    try {
      const results = await this.zrevrange(`leaderboard:${category}:${period}:${periodKey}`, 0, limit - 1, true);
      const traders = [];
      
      for (let i = 0; i < results.length; i += 2) {
        const userId = results[i];
        const score = parseFloat(results[i + 1]);
        
        traders.push({
          userId,
          score,
          rank: Math.floor(i / 2) + 1
        });
      }
      
      return traders;
    } catch (error) {
      console.error('Error getting period leaderboard:', error);
      return [];
    }
  }

  // Health Check
  // Paper Trading specific methods
  async addPaperTradingOrder(order: any): Promise<void> {
    try {
      await this.lpush('paper_trading_orders', order);
    } catch (error) {
      console.error('Error adding paper trading order:', error);
      throw error;
    }
  }

  async getNextPaperTradingOrder(): Promise<any> {
    try {
      return await this.rpop('paper_trading_orders');
    } catch (error) {
      console.error('Error getting next paper trading order:', error);
      return null;
    }
  }

  async subscribeToOrderUpdates(callback: (message: any) => void): Promise<void> {
    try {
      await this.subscribe('order_updates', callback);
    } catch (error) {
      console.error('Error subscribing to order updates:', error);
      throw error;
    }
  }

  async publishOrderUpdate(orderUpdate: any): Promise<void> {
    try {
      await this.publish('order_updates', orderUpdate);
    } catch (error) {
      console.error('Error publishing order update:', error);
      throw error;
    }
  }

  async cacheQueueStats(stats: { pending: number; processing: number; completed: number }): Promise<void> {
    try {
      await this.hset('queue_stats', 'pending', stats.pending.toString());
      await this.hset('queue_stats', 'processing', stats.processing.toString());
      await this.hset('queue_stats', 'completed', stats.completed.toString());
    } catch (error) {
      console.error('Error caching queue stats:', error);
      throw error;
    }
  }

  async getQueueStats(): Promise<{ pending: number; processing: number; completed: number }> {
    try {
      const stats = await this.hgetall('queue_stats');
      return {
        pending: parseInt(stats.pending || '0'),
        processing: parseInt(stats.processing || '0'),
        completed: parseInt(stats.completed || '0')
      };
    } catch (error) {
      console.error('Error getting queue stats:', error);
      return { pending: 0, processing: 0, completed: 0 };
    }
  }

  isHealthy(): boolean {
    return this.isConnected;
  }
}

// Export singleton instance
export const redisService = new RedisService();
export default redisService;