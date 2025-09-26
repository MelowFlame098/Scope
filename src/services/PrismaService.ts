import { 
  User, 
  Portfolio, 
  Trade, 
  Position, 
  Watchlist, 
  Alert,
  SocialProfile,
  Follow,
  SocialPost,
  Comment,
  Like,
  CopyTradingSettings,
  CopiedTrade,
  TradeSide,
  TradeType,
  TradeStatus,
  AlertCondition
} from '@prisma/client';
import { Decimal } from 'decimal.js';

// Mock Prisma client for development without database connection
const mockPrisma = {
  user: {
    create: async (data: any) => ({ id: 'mock-user-id', ...data.data }),
    findUnique: async (query: any) => null,
    findMany: async (query: any) => [],
    update: async (query: any) => ({ id: 'mock-user-id', ...query.data }),
  },
  portfolio: {
    create: async (data: any) => ({ id: 'mock-portfolio-id', ...data.data }),
    findMany: async (query: any) => [
      { 
        id: 'mock-portfolio-1', 
        name: 'Main Portfolio',
        description: 'Primary trading portfolio',
        totalValue: new Decimal(50000),
        cashBalance: new Decimal(10000),
        userId: 'user1',
        isDefault: true,
        isPaperTrade: false,
        createdAt: new Date(),
        updatedAt: new Date()
      }
    ],
    update: async (query: any) => ({ id: 'mock-portfolio-id', ...query.data }),
  },
  trade: {
    create: async (data: any) => ({ id: 'mock-trade-id', ...data.data }),
    findMany: async (query: any) => [
      { 
        id: 'mock-trade-1', 
        symbol: 'AAPL', 
        price: new Decimal(150), 
        executedPrice: new Decimal(155), 
        quantity: new Decimal(10),
        userId: 'user1',
        portfolioId: 'portfolio1',
        status: 'FILLED' as TradeStatus,
        type: 'BUY' as TradeType,
        side: 'BUY' as TradeSide,
        createdAt: new Date(),
        updatedAt: new Date(),
        executedAt: new Date(),
        executedQty: new Decimal(10),
        fees: new Decimal(2.5),
        stopLoss: null,
        takeProfit: null,
        leverage: null
      },
      { 
        id: 'mock-trade-2', 
        symbol: 'GOOGL', 
        price: new Decimal(2500), 
        executedPrice: new Decimal(2480), 
        quantity: new Decimal(5),
        userId: 'user1',
        portfolioId: 'portfolio1',
        status: 'FILLED' as TradeStatus,
        type: 'SELL' as TradeType,
        side: 'SELL' as TradeSide,
        createdAt: new Date(),
        updatedAt: new Date(),
        executedAt: new Date(),
        executedQty: new Decimal(5),
        fees: new Decimal(5.0),
        stopLoss: null,
        takeProfit: null,
        leverage: null
      }
    ],
    update: async (query: any) => ({ id: 'mock-trade-id', ...query.data }),
    count: async (query: any) => 2,
  },
  position: {
    upsert: async (query: any) => ({ id: 'mock-position-id', ...query.create }),
    findMany: async (query: any) => [],
  },
  watchlist: {
    create: async (data: any) => ({ id: 'mock-watchlist-id', ...data.data }),
    findMany: async (query: any) => [],
    update: async (query: any) => ({ id: 'mock-watchlist-id', ...query.data }),
    count: async (query: any) => 0,
  },
  alert: {
    create: async (data: any) => ({ id: 'mock-alert-id', ...data.data }),
    findMany: async (query: any) => [],
    update: async (query: any) => ({ id: 'mock-alert-id', ...query.data }),
    count: async (query: any) => 0,
  },
  socialProfile: {
    create: async (data: any) => ({ id: 'mock-profile-id', ...data.data }),
    update: async (query: any) => ({ id: 'mock-profile-id', ...query.data }),
    findMany: async (query: any) => [],
  },
  follow: {
    create: async (data: any) => ({ id: 'mock-follow-id', ...data.data }),
    delete: async (query: any) => ({ id: 'mock-follow-id' }),
    findMany: async (query: any) => [
      { id: 'mock-follow-1', followerId: 'user1', followingId: 'user2' },
      { id: 'mock-follow-2', followerId: 'user1', followingId: 'user3' }
    ],
    findUnique: async (query: any) => null,
  },
  socialPost: {
    create: async (data: any) => ({ id: 'mock-post-id', ...data.data }),
    findMany: async (query: any) => [],
    update: async (query: any) => ({ id: 'mock-post-id', ...query.data }),
  },
  like: {
    create: async (data: any) => ({ id: 'mock-like-id', ...data.data }),
    findUnique: async (query: any) => null,
    delete: async (query: any) => ({ id: 'mock-like-id' }),
  },
  copyTradingSettings: {
    create: async (data: any) => ({ id: 'mock-settings-id', ...data.data }),
    findMany: async (query: any) => [],
  },
  copiedTrade: {
    create: async (data: any) => ({ id: 'mock-copied-trade-id', ...data.data }),
  },
  $transaction: async (operations: any[]) => {
    return operations.map((op, index) => ({ id: `mock-transaction-${index}` }));
  },
};

export class PrismaService {
  // User Management
  static async createUser(data: {
    email: string;
    username: string;
    hashedPassword: string;
    firstName?: string;
    lastName?: string;
  }): Promise<User> {
    return await mockPrisma.user.create({
      data,
    });
  }

  static async getUserByEmail(email: string): Promise<User | null> {
    return await mockPrisma.user.findUnique({
      where: { email },
      include: {
        portfolios: true,
        socialProfile: true,
      },
    });
  }

  static async getUserById(id: string): Promise<User | null> {
    return await mockPrisma.user.findUnique({
      where: { id },
      include: {
        portfolios: {
          include: {
            positions: true,
            trades: {
              orderBy: { createdAt: 'desc' },
              take: 10,
            },
          },
        },
        socialProfile: true,
        watchlists: true,
        alerts: { where: { isActive: true } },
      },
    });
  }

  // Portfolio Management
  static async createPortfolio(data: {
    userId: string;
    name: string;
    description?: string;
    cashBalance?: number;
    isPaperTrade?: boolean;
  }): Promise<Portfolio> {
    return await mockPrisma.portfolio.create({
      data: {
        ...data,
        totalValue: data.cashBalance || 100000,
        cashBalance: data.cashBalance || 100000,
        isPaperTrade: data.isPaperTrade ?? true,
      },
    });
  }

  static async getUserPortfolios(userId: string): Promise<Portfolio[]> {
    return await mockPrisma.portfolio.findMany({
      where: { userId },
      include: {
        positions: true,
        trades: {
          orderBy: { createdAt: 'desc' },
          take: 5,
        },
      },
    });
  }

  static async updatePortfolioValue(portfolioId: string, totalValue: number): Promise<Portfolio> {
    return await mockPrisma.portfolio.update({
      where: { id: portfolioId },
      data: { totalValue },
    });
  }

  // Trading Operations
  static async createTrade(data: {
    userId: string;
    portfolioId: string;
    symbol: string;
    side: TradeSide;
    type: TradeType;
    quantity: number;
    price?: number;
  }): Promise<Trade> {
    return await mockPrisma.trade.create({
      data: {
        ...data,
        status: TradeStatus.PENDING,
      },
    });
  }

  static async updateTradeStatus(
    tradeId: string, 
    status: TradeStatus, 
    executedPrice?: number,
    executedQty?: number,
    fees?: number
  ): Promise<Trade> {
    return await mockPrisma.trade.update({
      where: { id: tradeId },
      data: {
        status,
        executedPrice,
        executedQty,
        fees,
        executedAt: status === TradeStatus.FILLED ? new Date() : undefined,
      },
    });
  }

  static async getUserTrades(userId: string, limit = 50): Promise<Trade[]> {
    return await mockPrisma.trade.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: limit,
    });
  }

  // Position Management
  static async upsertPosition(data: {
    portfolioId: string;
    symbol: string;
    quantity: number;
    averagePrice: number;
    currentPrice: number;
    marketValue: number;
    unrealizedPnL: number;
  }): Promise<Position> {
    return await mockPrisma.position.upsert({
      where: {
        portfolioId_symbol: {
          portfolioId: data.portfolioId,
          symbol: data.symbol,
        },
      },
      update: {
        quantity: data.quantity,
        averagePrice: data.averagePrice,
        currentPrice: data.currentPrice,
        marketValue: data.marketValue,
        unrealizedPnL: data.unrealizedPnL,
      },
      create: data,
    });
  }

  static async getPortfolioPositions(portfolioId: string): Promise<Position[]> {
    return await mockPrisma.position.findMany({
      where: { portfolioId },
    });
  }

  // Watchlist Management
  static async createWatchlist(data: {
    userId: string;
    name: string;
    symbols: string[];
  }): Promise<Watchlist> {
    return await mockPrisma.watchlist.create({
      data,
    });
  }

  static async getUserWatchlists(userId: string): Promise<Watchlist[]> {
    return await mockPrisma.watchlist.findMany({
      where: { userId },
    });
  }

  static async updateWatchlist(id: string, symbols: string[]): Promise<Watchlist> {
    return await mockPrisma.watchlist.update({
      where: { id },
      data: { symbols },
    });
  }

  // Alert Management
  static async createAlert(data: {
    userId: string;
    symbol: string;
    condition: AlertCondition;
    targetPrice: number;
    message?: string;
  }): Promise<Alert> {
    return await mockPrisma.alert.create({
      data,
    });
  }

  static async getUserAlerts(userId: string): Promise<Alert[]> {
    return await mockPrisma.alert.findMany({
      where: { userId },
    });
  }

  static async triggerAlert(alertId: string): Promise<Alert> {
    return await mockPrisma.alert.update({
      where: { id: alertId },
      data: {
        isTriggered: true,
        triggeredAt: new Date(),
      },
    });
  }

  // Social Trading - Profile Management
  static async createSocialProfile(data: {
    userId: string;
    displayName?: string;
    bio?: string;
    avatar?: string;
    isPublic?: boolean;
  }): Promise<SocialProfile> {
    return await mockPrisma.socialProfile.create({
      data,
    });
  }

  static async updateTradingStats(userId: string, stats: {
    totalReturn?: number;
    winRate?: number;
    tradesCount?: number;
  }): Promise<SocialProfile | null> {
    return await mockPrisma.socialProfile.update({
      where: { userId },
      data: stats,
    });
  }

  static async followTrader(followerId: string, followingId: string): Promise<Follow> {
    // Check if already following
    const existingFollow = await mockPrisma.follow.findUnique({
      where: {
        followerId_followingId: {
          followerId,
          followingId,
        },
      },
    });

    if (existingFollow) {
      throw new Error('Already following this trader');
    }

    return await mockPrisma.follow.create({
      data: {
        followerId,
        followingId,
      },
    });
  }

  static async unfollowTrader(followerId: string, followingId: string): Promise<void> {
    await mockPrisma.follow.delete({
      where: {
        followerId_followingId: {
          followerId,
          followingId,
        },
      },
    });
  }

  static async getTopTraders(limit = 20): Promise<(SocialProfile & { user: User })[]> {
    return await mockPrisma.socialProfile.findMany({
      where: { isPublic: true },
      orderBy: { totalReturn: 'desc' },
      take: limit,
      include: { user: true },
    });
  }

  // Social Posts
  static async createPost(data: {
    userId: string;
    content: string;
    imageUrl?: string;
  }): Promise<SocialPost> {
    return await mockPrisma.socialPost.create({
      data,
    });
  }

  static async getSocialFeed(userId: string, limit = 20): Promise<SocialPost[]> {
    // Get posts from followed users
    const following = await mockPrisma.follow.findMany({
      where: { followerId: userId },
      select: { followingId: true },
    });

    const followingIds = following.map(f => f.followingId);
    followingIds.push(userId); // Include user's own posts

    return await mockPrisma.socialPost.findMany({
      where: {
        userId: { in: followingIds },
      },
      orderBy: { createdAt: 'desc' },
      take: limit,
      include: {
        user: true,
        likes: true,
        _count: {
          select: { likes: true },
        },
      },
    });
  }

  static async likePost(postId: string, userId: string): Promise<Like> {
    // Check if already liked
    const existingLike = await mockPrisma.like.findUnique({
      where: {
        postId_userId: {
          postId,
          userId,
        },
      },
    });

    if (existingLike) {
      throw new Error('Post already liked');
    }

    return await mockPrisma.like.create({
      data: {
        postId,
        userId,
      },
    });
  }

  // Copy Trading
  static async createCopySettings(data: {
    userId: string;
    targetTraderId: string;
    copyRatio: number;
    maxRiskPercent: number;
  }): Promise<CopyTradingSettings> {
    return await mockPrisma.copyTradingSettings.create({
      data,
    });
  }

  static async getCopySettings(userId: string): Promise<CopyTradingSettings[]> {
    return await mockPrisma.copyTradingSettings.findMany({
      where: { userId },
    });
  }

  static async createCopiedTrade(data: {
    originalTradeId: string;
    userId: string;
    symbol: string;
    side: TradeSide;
    quantity: number;
    price: number;
  }): Promise<CopiedTrade> {
    return await mockPrisma.copiedTrade.create({
      data,
    });
  }

  // Analytics and Reporting
  static async getPortfolioPerformance(portfolioId: string, days = 30) {
    const trades = await mockPrisma.trade.findMany({
      where: {
        portfolioId,
        createdAt: {
          gte: new Date(Date.now() - days * 24 * 60 * 60 * 1000),
        },
      },
    });

    // Calculate performance metrics
    const totalTrades = trades.length;
    const winningTrades = trades.filter(t => (t.executedPrice || 0) > (t.price || 0)).length;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

    return {
      totalTrades,
      winRate,
      totalReturn: 0, // Mock value
      sharpeRatio: 0, // Mock value
    };
  }

  static async getUserStats(userId: string) {
    const tradesCount = await mockPrisma.trade.count({
      where: { userId },
    });

    const portfolios = await mockPrisma.portfolio.findMany({
      where: { userId },
    });

    const totalValue = portfolios.reduce((sum, p) => sum + (p.totalValue ? p.totalValue.toNumber() : 0), 0);

    return {
      tradesCount,
      portfoliosCount: portfolios.length,
      totalValue,
      totalReturn: 0, // Mock value
    };
  }
}

export default PrismaService;