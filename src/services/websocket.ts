import { Asset, NewsItem, ModelPrediction } from '../store/useStore';

// WebSocket Events
export interface WebSocketEvents {
  // Connection events
  connect: () => void;
  disconnect: (reason: string) => void;
  connect_error: (error: Error) => void;
  
  // Market data events
  market_update: (data: { type: string; data: Asset[]; timestamp: string }) => void;
  price_update: (data: { assetId: string; price: number; change: number }) => void;
  asset_update: (data: Asset) => void;
  
  // News events
  news_update: (data: { type: string; data: NewsItem; timestamp: string }) => void;
  breaking_news: (newsItem: NewsItem) => void;
  
  // AI events
  ai_insight: (data: { type: string; data: any; timestamp: string }) => void;
  model_prediction: (prediction: ModelPrediction) => void;
  
  // User events
  notification: (notification: any) => void;
  
  // System events
  system_alert: (alert: { type: 'info' | 'warning' | 'error'; message: string }) => void;
}

class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private eventListeners: Map<string, Function[]> = new Map();
  private subscriptions: Set<string> = new Set();
  private reconnectTimer: NodeJS.Timeout | null = null;
  
  constructor() {
    this.setupEventListeners();
  }
  
  // Initialize WebSocket connection
  connect(url?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      
      if (this.isConnecting) {
        reject(new Error('Connection already in progress'));
        return;
      }
      
      this.isConnecting = true;
      const wsUrl = url || process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
      
      try {
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
          }
          this.emit('connect');
          resolve();
        };
        
        this.socket.onclose = (event) => {
          console.log('WebSocket disconnected:', event.reason);
          this.isConnecting = false;
          this.emit('disconnect', event.reason || 'Connection closed');
          this.handleReconnect();
        };
        
        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.emit('connect_error', new Error('WebSocket connection failed'));
          reject(error);
        };
        
        this.socket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
      
      // Set up connection timeout
      setTimeout(() => {
        if (this.isConnecting) {
          this.isConnecting = false;
          reject(new Error('Connection timeout'));
        }
      }, 20000);
    });
  }
  
  // Disconnect WebSocket
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.subscriptions.clear();
    this.isConnecting = false;
  }
  
  // Check if connected
  isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN || false;
  }
  
  // Handle incoming messages
  private handleMessage(message: any): void {
    const { type, data, timestamp } = message;
    
    switch (type) {
      case 'market_update':
        this.emit('market_update', { type, data, timestamp });
        break;
      case 'price_update':
        this.emit('price_update', data);
        break;
      case 'asset_update':
        this.emit('asset_update', data);
        break;
      case 'news_update':
        this.emit('news_update', { type, data, timestamp });
        break;
      case 'breaking_news':
        this.emit('breaking_news', data);
        break;
      case 'ai_insight':
        this.emit('ai_insight', { type, data, timestamp });
        break;
      case 'model_prediction':
        this.emit('model_prediction', data);
        break;
      case 'notification':
        this.emit('notification', data);
        break;
      case 'system_alert':
        this.emit('system_alert', data);
        break;
      default:
        console.log('Unknown message type:', type);
    }
  }
  
  // Send message to server
  private sendMessage(message: any): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }
  
  // Subscribe to asset updates
  subscribeToAsset(assetId: string): void {
    this.sendMessage({
      type: 'subscribe_asset',
      data: { asset_id: assetId }
    });
    this.subscriptions.add(`asset:${assetId}`);
  }
  
  // Unsubscribe from asset updates
  unsubscribeFromAsset(assetId: string): void {
    this.sendMessage({
      type: 'unsubscribe_asset',
      data: { asset_id: assetId }
    });
    this.subscriptions.delete(`asset:${assetId}`);
  }
  
  // Subscribe to news category
  subscribeToNews(category?: string): void {
    this.sendMessage({
      type: 'subscribe_news',
      data: { category }
    });
    this.subscriptions.add(`news:${category || 'all'}`);
  }
  
  // Subscribe to model updates
  subscribeToModel(modelId: string): void {
    this.sendMessage({
      type: 'subscribe_model',
      data: { model_id: modelId }
    });
    this.subscriptions.add(`model:${modelId}`);
  }
  
  // Subscribe to forum updates
  subscribeToForum(category?: string): void {
    this.sendMessage({
      type: 'subscribe_forum',
      data: { category }
    });
    this.subscriptions.add(`forum:${category || 'all'}`);
  }
  
  // Subscribe to user-specific events
  subscribeToUser(): void {
    this.sendMessage({
      type: 'subscribe_user',
      data: {}
    });
    this.subscriptions.add('user');
  }
  
  // Run model via WebSocket
  runModel(modelId: string, assetId: string, parameters?: any): void {
    this.sendMessage({
      type: 'run_model',
      data: {
        model_id: modelId,
        asset_id: assetId,
        parameters
      }
    });
  }
  
  // Send message to forum
  sendForumMessage(data: any): void {
    this.sendMessage({
      type: 'forum_message',
      data
    });
  }
  
  // Add event listener
  on<K extends keyof WebSocketEvents>(event: K, listener: WebSocketEvents[K]): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(listener);
  }
  
  // Remove event listener
  off<K extends keyof WebSocketEvents>(event: K, listener: WebSocketEvents[K]): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }
  
  // Emit event to listeners
  private emit(event: string, ...args: any[]): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(...args);
        } catch (error) {
          console.error(`Error in WebSocket event listener for ${event}:`, error);
        }
      });
    }
  }
  
  // Setup event listeners
  private setupEventListeners(): void {
    // This method can be used to set up default event listeners
  }
  
  // Setup data event listeners
  private setupDataEventListeners(): void {
    // Event listeners are now handled in the handleMessage method
    // This method is kept for compatibility but no longer needed
  }
  
  // Handle reconnection logic
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts && !this.reconnectTimer) {
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
      this.reconnectTimer = setTimeout(() => {
        if (this.socket?.readyState !== WebSocket.OPEN) {
          this.reconnectAttempts++;
          console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          this.reconnectTimer = null;
          this.connect();
        }
      }, delay);
    }
  }
  
  // Resubscribe to all active subscriptions
  private resubscribeAll(): void {
    this.subscriptions.forEach(subscription => {
      const [type, id] = subscription.split(':');
      
      switch (type) {
        case 'asset':
          this.subscribeToAsset(id);
          break;
        case 'news':
          this.subscribeToNews(id === 'all' ? undefined : id);
          break;
        case 'model':
          this.subscribeToModel(id);
          break;
        case 'forum':
          this.subscribeToForum(id === 'all' ? undefined : id);
          break;
        case 'user':
          this.subscribeToUser();
          break;
      }
    });
  }
  
  // Get connection status and info
  getStatus(): {
    connected: boolean;
    reconnectAttempts: number;
    subscriptions: string[];
  } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      subscriptions: Array.from(this.subscriptions)
    };
  }
  
  // Update authentication token
  updateAuth(token: string): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.sendMessage({
        type: 'auth',
        data: { token }
      });
    }
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

// Export service and types
export { webSocketService, WebSocketService };
export default webSocketService;

// React hook for WebSocket
export const useWebSocket = () => {
  const connect = (url?: string) => webSocketService.connect(url);
  const disconnect = () => webSocketService.disconnect();
  const isConnected = () => webSocketService.isConnected();
  const getStatus = () => webSocketService.getStatus();
  
  const subscribeToAsset = (assetId: string) => webSocketService.subscribeToAsset(assetId);
  const unsubscribeFromAsset = (assetId: string) => webSocketService.unsubscribeFromAsset(assetId);
  const subscribeToNews = (category?: string) => webSocketService.subscribeToNews(category);
  const subscribeToModel = (modelId: string) => webSocketService.subscribeToModel(modelId);
  const subscribeToForum = (category?: string) => webSocketService.subscribeToForum(category);
  const subscribeToUser = () => webSocketService.subscribeToUser();
  
  const runModel = (modelId: string, assetId: string, parameters?: any) => 
    webSocketService.runModel(modelId, assetId, parameters);
  
  const on = <K extends keyof WebSocketEvents>(event: K, listener: WebSocketEvents[K]) => 
    webSocketService.on(event, listener);
  
  const off = <K extends keyof WebSocketEvents>(event: K, listener: WebSocketEvents[K]) => 
    webSocketService.off(event, listener);
  
  return {
    connect,
    disconnect,
    isConnected,
    getStatus,
    subscribeToAsset,
    unsubscribeFromAsset,
    subscribeToNews,
    subscribeToModel,
    subscribeToForum,
    subscribeToUser,
    runModel,
    on,
    off
  };
};