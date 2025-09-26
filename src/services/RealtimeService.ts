import { EventEmitter } from 'events';
import { redisService } from './RedisService';
import { marketDataService, PriceData } from './MarketDataService';
import { newsService, NewsArticle } from './NewsService';

export interface RealtimeNotification {
  id: string;
  type: 'price_alert' | 'news_alert' | 'trade_executed' | 'system_alert' | 'user_message';
  title: string;
  message: string;
  data?: any;
  userId?: string;
  timestamp: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  read: boolean;
}

export interface PriceAlert {
  id: string;
  userId: string;
  symbol: string;
  condition: 'above' | 'below' | 'change_percent';
  targetValue: number;
  currentValue?: number;
  isActive: boolean;
  createdAt: number;
  triggeredAt?: number;
}

export interface UserPresence {
  userId: string;
  status: 'online' | 'away' | 'busy' | 'offline';
  lastSeen: number;
  currentPage?: string;
  tradingSession?: {
    startTime: number;
    activeSymbols: string[];
    totalTrades: number;
  };
}

class RealtimeService extends EventEmitter {
  private connectedUsers: Map<string, UserPresence> = new Map();
  private userAlerts: Map<string, PriceAlert[]> = new Map();
  private activeSubscriptions: Set<string> = new Set();
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.initializeRealtimeChannels();
    this.startHeartbeat();
  }

  private initializeRealtimeChannels() {
    // Subscribe to market data updates
    this.subscribeToChannel('market:updates', (data: any) => {
      this.handleMarketUpdate(data);
    });

    // Subscribe to news updates
    this.subscribeToChannel('news:breaking', (data: NewsArticle) => {
      this.handleBreakingNews(data);
    });

    // Subscribe to system notifications
    this.subscribeToChannel('system:notifications', (data: RealtimeNotification) => {
      this.handleSystemNotification(data);
    });

    // Subscribe to user presence updates
    this.subscribeToChannel('user:presence', (data: UserPresence) => {
      this.handlePresenceUpdate(data);
    });

    console.log('Realtime service initialized with Redis pub/sub');
  }

  private async subscribeToChannel(channel: string, callback: (data: any) => void) {
    try {
      await redisService.subscribe(channel, callback);
      this.activeSubscriptions.add(channel);
      console.log(`Subscribed to channel: ${channel}`);
    } catch (error) {
      console.error(`Failed to subscribe to channel ${channel}:`, error);
    }
  }

  private startHeartbeat() {
    // Send heartbeat every 30 seconds to maintain connections
    this.heartbeatInterval = setInterval(() => {
      this.publishHeartbeat();
    }, 30000);
  }

  private async publishHeartbeat() {
    const heartbeat = {
      timestamp: Date.now(),
      activeUsers: this.connectedUsers.size,
      activeSubscriptions: this.activeSubscriptions.size
    };

    await redisService.publish('system:heartbeat', heartbeat);
  }

  // Market Data Real-time Updates
  private handleMarketUpdate(data: any) {
    if (data.type === 'price_update') {
      this.checkPriceAlerts(data.symbol, data.price);
      this.emit('priceUpdate', data);
    }
  }

  async publishPriceUpdate(symbol: string, priceData: PriceData) {
    const update = {
      type: 'price_update',
      symbol,
      price: priceData.price,
      change: priceData.change,
      changePercent: priceData.changePercent,
      volume: priceData.volume,
      timestamp: priceData.timestamp
    };

    await redisService.publish('market:updates', update);
    await redisService.publish(`price:${symbol}`, update);
  }

  // News Real-time Updates
  private handleBreakingNews(article: NewsArticle) {
    // Create notification for breaking news
    const notification: RealtimeNotification = {
      id: `news_${article.id}`,
      type: 'news_alert',
      title: 'Breaking News',
      message: article.title,
      data: article,
      timestamp: Date.now(),
      priority: article.impact === 'high' ? 'critical' : 'high',
      read: false
    };

    this.broadcastNotification(notification);
  }

  async publishBreakingNews(article: NewsArticle) {
    await redisService.publish('news:breaking', article);
    
    // Also cache the breaking news
    await redisService.lpush('news:breaking:recent', article);
    
    // Keep only last 50 breaking news items
    const recentNews = await redisService.lrange('news:breaking:recent', 0, 49);
    if (recentNews.length > 50) {
      await redisService.del('news:breaking:recent');
      for (const news of recentNews.slice(0, 50)) {
        await redisService.lpush('news:breaking:recent', news);
      }
    }
  }

  // Price Alerts Management
  async createPriceAlert(alert: Omit<PriceAlert, 'id' | 'createdAt'>): Promise<string> {
    const alertId = `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const fullAlert: PriceAlert = {
      ...alert,
      id: alertId,
      createdAt: Date.now()
    };

    // Store in Redis
    await redisService.hset(`alerts:${alert.userId}`, alertId, fullAlert);
    
    // Store in memory for quick access
    const userAlerts = this.userAlerts.get(alert.userId) || [];
    userAlerts.push(fullAlert);
    this.userAlerts.set(alert.userId, userAlerts);

    return alertId;
  }

  async getUserAlerts(userId: string): Promise<PriceAlert[]> {
    // Try memory cache first
    const memoryAlerts = this.userAlerts.get(userId);
    if (memoryAlerts) {
      return memoryAlerts.filter(alert => alert.isActive);
    }

    // Load from Redis
    const alertsData = await redisService.hgetall<PriceAlert>(`alerts:${userId}`);
    const alerts = Object.values(alertsData).filter(alert => alert.isActive);
    
    // Cache in memory
    this.userAlerts.set(userId, alerts);
    
    return alerts;
  }

  async deleteAlert(userId: string, alertId: string): Promise<boolean> {
    // Remove from Redis
    await redisService.hdel(`alerts:${userId}`, alertId);
    
    // Remove from memory
    const userAlerts = this.userAlerts.get(userId) || [];
    const updatedAlerts = userAlerts.filter(alert => alert.id !== alertId);
    this.userAlerts.set(userId, updatedAlerts);

    return true;
  }

  private async checkPriceAlerts(symbol: string, currentPrice: number) {
    // Check all active alerts for this symbol
    const userIds = Array.from(this.userAlerts.keys());
    for (const userId of userIds) {
      const alerts = this.userAlerts.get(userId) || [];
      const symbolAlerts = alerts.filter(alert => 
        alert.symbol === symbol && alert.isActive
      );

      for (const alert of symbolAlerts) {
        let shouldTrigger = false;

        switch (alert.condition) {
          case 'above':
            shouldTrigger = currentPrice >= alert.targetValue;
            break;
          case 'below':
            shouldTrigger = currentPrice <= alert.targetValue;
            break;
          case 'change_percent':
            // This would require historical data to calculate
            // For now, we'll skip this condition
            break;
        }

        if (shouldTrigger) {
          await this.triggerPriceAlert(alert, currentPrice);
        }
      }
    }
  }

  private async triggerPriceAlert(alert: PriceAlert, currentPrice: number) {
    // Mark alert as triggered
    alert.isActive = false;
    alert.triggeredAt = Date.now();
    alert.currentValue = currentPrice;

    // Update in Redis
    await redisService.hset(`alerts:${alert.userId}`, alert.id, alert);

    // Create notification
    const notification: RealtimeNotification = {
      id: `alert_${alert.id}`,
      type: 'price_alert',
      title: 'Price Alert Triggered',
      message: `${alert.symbol} has reached ${currentPrice} (target: ${alert.targetValue})`,
      data: alert,
      userId: alert.userId,
      timestamp: Date.now(),
      priority: 'high',
      read: false
    };

    await this.sendNotificationToUser(alert.userId, notification);
  }

  // User Presence Management
  async updateUserPresence(userId: string, presence: Partial<UserPresence>) {
    const currentPresence = this.connectedUsers.get(userId) || {
      userId,
      status: 'offline',
      lastSeen: Date.now()
    };

    const updatedPresence: UserPresence = {
      ...currentPresence,
      ...presence,
      lastSeen: Date.now()
    };

    // Update in memory
    this.connectedUsers.set(userId, updatedPresence);

    // Update in Redis for persistence
    await redisService.hset('user:presence', userId, updatedPresence);

    // Broadcast presence update
    await redisService.publish('user:presence', updatedPresence);

    return updatedPresence;
  }

  async getUserPresence(userId: string): Promise<UserPresence | null> {
    // Try memory first
    const memoryPresence = this.connectedUsers.get(userId);
    if (memoryPresence) {
      return memoryPresence;
    }

    // Try Redis
    const redisPresence = await redisService.hget<UserPresence>('user:presence', userId);
    if (redisPresence) {
      this.connectedUsers.set(userId, redisPresence);
      return redisPresence;
    }

    return null;
  }

  async getOnlineUsers(): Promise<UserPresence[]> {
    const allPresence = await redisService.hgetall<UserPresence>('user:presence');
    const onlineUsers = Object.values(allPresence).filter(
      presence => presence.status === 'online' && 
      (Date.now() - presence.lastSeen) < 300000 // 5 minutes
    );

    return onlineUsers;
  }

  private handlePresenceUpdate(presence: UserPresence) {
    this.connectedUsers.set(presence.userId, presence);
    this.emit('presenceUpdate', presence);
  }

  // Notifications Management
  async sendNotificationToUser(userId: string, notification: RealtimeNotification) {
    // Store notification in Redis
    await redisService.lpush(`notifications:${userId}`, notification);
    
    // Keep only last 100 notifications per user
    const notifications = await redisService.lrange(`notifications:${userId}`, 0, 99);
    if (notifications.length > 100) {
      await redisService.del(`notifications:${userId}`);
      for (const notif of notifications.slice(0, 100)) {
        await redisService.lpush(`notifications:${userId}`, notif);
      }
    }

    // Publish to user's personal channel
    await redisService.publish(`user:${userId}:notifications`, notification);

    this.emit('notification', { userId, notification });
  }

  async broadcastNotification(notification: RealtimeNotification) {
    // Broadcast to all connected users
    await redisService.publish('system:notifications', notification);
    
    // Store in global notifications
    await redisService.lpush('notifications:global', notification);

    this.emit('broadcast', notification);
  }

  async getUserNotifications(userId: string, limit: number = 20): Promise<RealtimeNotification[]> {
    const notifications = await redisService.lrange<RealtimeNotification>(
      `notifications:${userId}`, 0, limit - 1
    );
    return notifications;
  }

  async markNotificationAsRead(userId: string, notificationId: string): Promise<boolean> {
    const notifications = await redisService.lrange<RealtimeNotification>(
      `notifications:${userId}`, 0, -1
    );
    
    const updatedNotifications = notifications.map(notif => 
      notif.id === notificationId ? { ...notif, read: true } : notif
    );

    // Replace the entire list
    await redisService.del(`notifications:${userId}`);
    for (const notif of updatedNotifications.reverse()) {
      await redisService.lpush(`notifications:${userId}`, notif);
    }

    return true;
  }

  // System Notifications
  private handleSystemNotification(notification: RealtimeNotification) {
    this.emit('systemNotification', notification);
  }

  // Trading Session Management
  async startTradingSession(userId: string, symbols: string[] = []) {
    const presence = await this.getUserPresence(userId);
    if (presence) {
      presence.tradingSession = {
        startTime: Date.now(),
        activeSymbols: symbols,
        totalTrades: 0
      };
      await this.updateUserPresence(userId, presence);
    }
  }

  async updateTradingSession(userId: string, update: Partial<UserPresence['tradingSession']>) {
    const presence = await this.getUserPresence(userId);
    if (presence && presence.tradingSession) {
      presence.tradingSession = { ...presence.tradingSession, ...update };
      await this.updateUserPresence(userId, presence);
    }
  }

  async endTradingSession(userId: string) {
    const presence = await this.getUserPresence(userId);
    if (presence) {
      delete presence.tradingSession;
      await this.updateUserPresence(userId, presence);
    }
  }

  // Connection Management
  async connectUser(userId: string, metadata?: any) {
    await this.updateUserPresence(userId, {
      status: 'online',
      ...metadata
    });

    // Subscribe to user's personal notification channel
    await this.subscribeToChannel(`user:${userId}:notifications`, (notification) => {
      this.emit('userNotification', { userId, notification });
    });

    console.log(`User ${userId} connected to realtime service`);
  }

  async disconnectUser(userId: string) {
    await this.updateUserPresence(userId, {
      status: 'offline'
    });

    this.connectedUsers.delete(userId);
    console.log(`User ${userId} disconnected from realtime service`);
  }

  // Health and Statistics
  getConnectionStats() {
    return {
      connectedUsers: this.connectedUsers.size,
      activeSubscriptions: this.activeSubscriptions.size,
      totalAlerts: Array.from(this.userAlerts.values()).reduce((sum, alerts) => sum + alerts.length, 0)
    };
  }

  async getSystemHealth() {
    const redisHealthy = redisService.isHealthy();
    const stats = this.getConnectionStats();
    
    return {
      status: redisHealthy ? 'healthy' : 'degraded',
      redis: redisHealthy,
      ...stats,
      timestamp: Date.now()
    };
  }

  // Cleanup
  disconnect() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    this.connectedUsers.clear();
    this.userAlerts.clear();
    this.activeSubscriptions.clear();
    
    this.emit('disconnected');
  }
}

// Export singleton instance
export const realtimeService = new RealtimeService();
export default RealtimeService;