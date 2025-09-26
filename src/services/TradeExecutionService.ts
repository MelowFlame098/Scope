import { redisService } from './RedisService';

export interface TradeOrder {
  id: string;
  userId: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'DAY';
  status: 'pending' | 'partial' | 'filled' | 'cancelled' | 'rejected';
  createdAt: Date;
  updatedAt: Date;
  executedQuantity: number;
  executedPrice?: number;
  fees: number;
  portfolioId?: string;
  isPaperTrade: boolean;
}

export interface TradeExecution {
  id: string;
  orderId: string;
  symbol: string;
  quantity: number;
  price: number;
  side: 'buy' | 'sell';
  timestamp: Date;
  fees: number;
  isPaperTrade: boolean;
}

export interface OrderBookEntry {
  price: number;
  quantity: number;
  timestamp: Date;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  lastUpdated: Date;
}

class TradeExecutionService {
  private readonly ORDERS_QUEUE = 'trade_orders';
  private readonly EXECUTIONS_QUEUE = 'trade_executions';
  private readonly ORDER_BOOK_PREFIX = 'orderbook:';
  private readonly USER_ORDERS_PREFIX = 'user_orders:';
  private readonly PENDING_ORDERS_SET = 'pending_orders';

  // Submit a new trade order
  async submitOrder(order: Omit<TradeOrder, 'id' | 'createdAt' | 'updatedAt' | 'executedQuantity' | 'fees'>): Promise<string> {
    const orderId = `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const fullOrder: TradeOrder = {
      ...order,
      id: orderId,
      createdAt: new Date(),
      updatedAt: new Date(),
      executedQuantity: 0,
      fees: 0,
      status: 'pending'
    };

    // Store order in Redis
    await redisService.hset(`order:${orderId}`, {
      ...fullOrder,
      createdAt: fullOrder.createdAt.toISOString(),
      updatedAt: fullOrder.updatedAt.toISOString()
    });

    // Add to user's orders list
    await redisService.sadd(`${this.USER_ORDERS_PREFIX}${order.userId}`, orderId);

    // Add to pending orders set with priority (timestamp)
    await redisService.zadd(this.PENDING_ORDERS_SET, Date.now(), orderId);

    // Queue order for processing
    await redisService.lpush(this.ORDERS_QUEUE, JSON.stringify(fullOrder));

    return orderId;
  }

  // Get order by ID
  async getOrder(orderId: string): Promise<TradeOrder | null> {
    const orderData = await redisService.hgetall(`order:${orderId}`);
    if (!orderData || Object.keys(orderData).length === 0) {
      return null;
    }

    return {
      ...orderData,
      quantity: parseFloat(orderData.quantity),
      price: orderData.price ? parseFloat(orderData.price) : undefined,
      stopPrice: orderData.stopPrice ? parseFloat(orderData.stopPrice) : undefined,
      executedQuantity: parseFloat(orderData.executedQuantity),
      fees: parseFloat(orderData.fees),
      createdAt: new Date(orderData.createdAt),
      updatedAt: new Date(orderData.updatedAt),
      isPaperTrade: orderData.isPaperTrade === 'true'
    } as TradeOrder;
  }

  // Get user's orders
  async getUserOrders(userId: string, limit: number = 50): Promise<TradeOrder[]> {
    const orderIds = await redisService.smembers(`${this.USER_ORDERS_PREFIX}${userId}`);
    const orders: TradeOrder[] = [];

    for (const orderId of orderIds.slice(0, limit)) {
      const order = await this.getOrder(orderId);
      if (order) {
        orders.push(order);
      }
    }

    // Sort by creation date (newest first)
    return orders.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  // Cancel an order
  async cancelOrder(orderId: string, userId: string): Promise<boolean> {
    const order = await this.getOrder(orderId);
    if (!order || order.userId !== userId) {
      return false;
    }

    if (order.status === 'filled' || order.status === 'cancelled') {
      return false;
    }

    // Update order status
    await redisService.hset(`order:${orderId}`, {
      status: 'cancelled',
      updatedAt: new Date().toISOString()
    });

    // Remove from pending orders
    await redisService.zrem(this.PENDING_ORDERS_SET, orderId);

    return true;
  }

  // Process pending orders (called by background worker)
  async processPendingOrders(): Promise<void> {
    // Get next batch of pending orders
    const orderIds = await redisService.zrange(this.PENDING_ORDERS_SET, 0, 9); // Process 10 at a time

    for (const orderId of orderIds) {
      try {
        await this.processOrder(orderId);
      } catch (error) {
        console.error(`Failed to process order ${orderId}:`, error);
      }
    }
  }

  // Process a single order
  private async processOrder(orderId: string): Promise<void> {
    const order = await this.getOrder(orderId);
    if (!order || order.status !== 'pending') {
      await redisService.zrem(this.PENDING_ORDERS_SET, orderId);
      return;
    }

    // For paper trading, simulate execution
    if (order.isPaperTrade) {
      await this.simulateExecution(order);
    } else {
      // For real trading, implement actual broker integration
      await this.executeRealOrder(order);
    }
  }

  // Simulate order execution for paper trading
  private async simulateExecution(order: TradeOrder): Promise<void> {
    // Get current market price (from market data service)
    const marketPrice = await this.getCurrentPrice(order.symbol);
    
    let executionPrice = marketPrice;
    let shouldExecute = false;

    // Determine if order should execute based on type
    switch (order.type) {
      case 'market':
        shouldExecute = true;
        break;
      case 'limit':
        if (order.side === 'buy' && order.price! >= marketPrice) {
          shouldExecute = true;
          executionPrice = Math.min(order.price!, marketPrice);
        } else if (order.side === 'sell' && order.price! <= marketPrice) {
          shouldExecute = true;
          executionPrice = Math.max(order.price!, marketPrice);
        }
        break;
      case 'stop':
        if (order.side === 'buy' && marketPrice >= order.stopPrice!) {
          shouldExecute = true;
        } else if (order.side === 'sell' && marketPrice <= order.stopPrice!) {
          shouldExecute = true;
        }
        break;
    }

    if (shouldExecute) {
      await this.executeOrder(order, executionPrice, order.quantity);
    }
  }

  // Execute order (update status and create execution record)
  private async executeOrder(order: TradeOrder, price: number, quantity: number): Promise<void> {
    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const fees = this.calculateFees(price * quantity, order.isPaperTrade);

    const execution: TradeExecution = {
      id: executionId,
      orderId: order.id,
      symbol: order.symbol,
      quantity,
      price,
      side: order.side,
      timestamp: new Date(),
      fees,
      isPaperTrade: order.isPaperTrade
    };

    // Store execution
    await redisService.hset(`execution:${executionId}`, {
      ...execution,
      timestamp: execution.timestamp.toISOString()
    });

    // Update order
    const newExecutedQuantity = order.executedQuantity + quantity;
    const newStatus = newExecutedQuantity >= order.quantity ? 'filled' : 'partial';

    await redisService.hset(`order:${order.id}`, {
      status: newStatus,
      executedQuantity: newExecutedQuantity.toString(),
      executedPrice: price.toString(),
      fees: (order.fees + fees).toString(),
      updatedAt: new Date().toISOString()
    });

    // Remove from pending if fully filled
    if (newStatus === 'filled') {
      await redisService.zrem(this.PENDING_ORDERS_SET, order.id);
    }

    // Queue execution for further processing (portfolio updates, notifications, etc.)
    await redisService.lpush(this.EXECUTIONS_QUEUE, JSON.stringify(execution));
  }

  // Get current market price (placeholder - integrate with market data service)
  private async getCurrentPrice(symbol: string): Promise<number> {
    // This would integrate with your market data service
    // For now, return a mock price
    const cachedPrice = await redisService.get(`price:${symbol}`);
    return cachedPrice ? parseFloat(cachedPrice) : 100; // Default price
  }

  // Calculate trading fees
  private calculateFees(notionalValue: number, isPaperTrade: boolean): number {
    if (isPaperTrade) return 0; // No fees for paper trading
    
    // Example fee structure: 0.1% of notional value, minimum $1
    const feeRate = 0.001;
    const calculatedFee = notionalValue * feeRate;
    return Math.max(calculatedFee, 1.0);
  }

  // Placeholder for real order execution
  private async executeRealOrder(order: TradeOrder): Promise<void> {
    // This would integrate with actual broker APIs
    // For now, just simulate
    await this.simulateExecution(order);
  }

  // Get order book for a symbol
  async getOrderBook(symbol: string): Promise<OrderBook | null> {
    const orderBookData = await redisService.hgetall(`${this.ORDER_BOOK_PREFIX}${symbol}`);
    if (!orderBookData || Object.keys(orderBookData).length === 0) {
      return null;
    }

    return {
      symbol,
      bids: JSON.parse(orderBookData.bids || '[]'),
      asks: JSON.parse(orderBookData.asks || '[]'),
      lastUpdated: new Date(orderBookData.lastUpdated)
    };
  }

  // Update order book (called by market data service)
  async updateOrderBook(symbol: string, bids: OrderBookEntry[], asks: OrderBookEntry[]): Promise<void> {
    await redisService.hset(`${this.ORDER_BOOK_PREFIX}${symbol}`, {
      bids: JSON.stringify(bids),
      asks: JSON.stringify(asks),
      lastUpdated: new Date().toISOString()
    });
  }

  // Get trade executions for a user
  async getUserExecutions(userId: string, limit: number = 50): Promise<TradeExecution[]> {
    const userOrders = await this.getUserOrders(userId, 100);
    const executions: TradeExecution[] = [];

    for (const order of userOrders) {
      if (order.status === 'filled' || order.status === 'partial') {
        // Get executions for this order
        const executionData = await redisService.hgetall(`execution:${order.id}`);
        if (executionData && Object.keys(executionData).length > 0) {
          executions.push({
            ...executionData,
            quantity: parseFloat(executionData.quantity),
            price: parseFloat(executionData.price),
            fees: parseFloat(executionData.fees),
            timestamp: new Date(executionData.timestamp),
            isPaperTrade: executionData.isPaperTrade === 'true'
          } as TradeExecution);
        }
      }
    }

    return executions
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  // Get queue statistics
  async getQueueStats(): Promise<{
    pending: number;
    processing: number;
    completed: number;
  }> {
    try {
      // Get stats from Redis
      const redisStats = await redisService.getQueueStats();
      
      // Also get current queue lengths for real-time data
      const pendingOrdersCount = await redisService.llen(this.ORDERS_QUEUE) || 0;
      const pendingExecutionsCount = await redisService.llen(this.EXECUTIONS_QUEUE) || 0;
      
      return {
        pending: Math.max(redisStats.pending, pendingOrdersCount),
        processing: redisStats.processing,
        completed: redisStats.completed
      };
    } catch (error) {
      console.error('Error getting queue stats:', error);
      return { pending: 0, processing: 0, completed: 0 };
    }
  }

  // Subscribe to real-time order updates
  async subscribeToOrderUpdates(callback: (update: any) => void): Promise<void> {
    try {
      await redisService.subscribeToOrderUpdates(callback);
    } catch (error) {
      console.error('Error subscribing to order updates:', error);
      throw error;
    }
  }

  // Update queue statistics
  async updateQueueStats(stats: { pending: number; processing: number; completed: number }): Promise<void> {
    try {
      await redisService.cacheQueueStats(stats);
    } catch (error) {
      console.error('Error updating queue stats:', error);
    }
  }
}

export const tradeExecutionService = new TradeExecutionService();