import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { io, Socket } from 'socket.io-client';

// Types
export interface Asset {
  id: string;
  symbol: string;
  name: string;
  category: 'crypto' | 'stocks' | 'forex' | 'commodities';
  price: number;
  change24h: number;
  volume: number;
  marketCap?: number;
  lastUpdated: string;
  isFavorite?: boolean;
}

export interface Model {
  id: string;
  name: string;
  category: 'crypto' | 'stocks' | 'forex' | 'cross-asset';
  type: 'technical' | 'fundamental' | 'ml' | 'sentiment' | 'statistical';
  description: string;
  accuracy?: number;
  lastRun?: string;
  isActive: boolean;
  status: 'idle' | 'running' | 'completed' | 'error';
}

export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  content: string;
  source: string;
  category: 'crypto' | 'stocks' | 'forex' | 'general' | 'defi' | 'nft';
  sentiment: 'positive' | 'neutral' | 'negative';
  publishedAt: string;
  url: string;
  imageUrl?: string;
  relevantAssets: string[];
}

export interface ForumPost {
  id: string;
  title: string;
  content: string;
  author: string;
  category: 'discussion' | 'analysis' | 'news' | 'strategy' | 'education';
  tags: string[];
  upvotes: number;
  downvotes: number;
  comments: number;
  createdAt: string;
  isSticky?: boolean;
  isTrending?: boolean;
}

export interface User {
  id: string;
  username: string;
  email: string;
  avatar?: string;
  preferences: {
    theme: 'light' | 'dark';
    defaultView: 'dashboard' | 'news' | 'research' | 'forum';
    notifications: boolean;
    autoRefresh: boolean;
  };
  subscription: 'free' | 'premium' | 'basic';
}

export interface ChartData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ModelPrediction {
  modelId: string;
  assetId: string;
  prediction: {
    direction: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    targetPrice?: number;
    timeframe: '1h' | '4h' | '1d' | '1w' | '1m';
    reasoning: string;
  };
  timestamp: string;
}

// Store State Interface
interface StoreState {
  // UI State
  activeView: 'dashboard' | 'portfolio' | 'trading' | 'charting' | 'risk' | 'analytics' | 'news' | 'research' | 'forum' | 'notifications' | 'subscription';
  selectedAssets: Asset[];
  selectedModels: Model[];
  isLoading: boolean;
  error: string | null;
  
  // Data
  assets: Asset[];
  models: Model[];
  news: NewsItem[];
  forumPosts: ForumPost[];
  user: User | null;
  chartData: Record<string, ChartData[]>;
  modelPredictions: ModelPrediction[];
  aiInsights: any[];
  watchlist: Asset[];
  notifications: Array<{id: string; message: string; type: string; read: boolean; timestamp: string;}>;
  
  // WebSocket
  socket: Socket | null;
  isConnected: boolean;
  
  // Search & Filters
  searchQuery: string;
  newsFilters: {
    category: string;
    sentiment: string;
    dateRange: string;
  };
  assetFilters: {
    category: string;
    sortBy: string;
    sortOrder: 'asc' | 'desc';
  };
  
  // Actions
  setActiveView: (view: 'dashboard' | 'portfolio' | 'trading' | 'charting' | 'risk' | 'analytics' | 'news' | 'research' | 'forum' | 'notifications' | 'subscription') => void;
  addSelectedAsset: (asset: Asset) => void;
  removeSelectedAsset: (assetId: string) => void;
  addSelectedModel: (model: Model) => void;
  removeSelectedModel: (modelId: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Data Actions
  updateAssets: (assets: Asset[]) => void;
  setAssets: (assets: Asset[]) => void;
  updateAsset: (asset: Asset) => void;
  updateModels: (models: Model[]) => void;
  updateNews: (news: NewsItem[]) => void;
  updateForumPosts: (posts: ForumPost[]) => void;
  setUser: (user: User | null) => void;
  updateChartData: (assetId: string, data: ChartData[]) => void;
  setChartData: (assetId: string, data: ChartData[]) => void;
  addModelPrediction: (prediction: ModelPrediction) => void;
  addAIInsight: (insight: any) => void;
  
  // Watchlist Actions
  addToWatchlist: (asset: Asset) => void;
  removeFromWatchlist: (assetId: string) => void;
  
  // Notification Actions
  addNotification: (notification: {id: string; message: string; type: string; read: boolean; timestamp: string;}) => void;
  markNotificationAsRead: (notificationId: string) => void;
  clearNotifications: () => void;
  
  // WebSocket Actions
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;
  
  // Search & Filter Actions
  setSearchQuery: (query: string) => void;
  setNewsFilters: (filters: Partial<StoreState['newsFilters']>) => void;
  setAssetFilters: (filters: Partial<StoreState['assetFilters']>) => void;
  
  // API Actions
  fetchAssets: () => Promise<void>;
  fetchModels: () => Promise<void>;
  fetchNews: () => Promise<void>;
  fetchForumPosts: () => Promise<void>;
  fetchChartData: (assetId: string, timeframe: string) => Promise<void>;
  runModel: (modelId: string, assetId: string) => Promise<void>;
  stopModel: (modelId: string) => Promise<void>;
}

// API Base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

// Create Store
export const useStore = create<StoreState>()
  (devtools(
    persist(
      (set, get) => ({
        // Initial State
        activeView: 'dashboard',
        selectedAssets: [],
        selectedModels: [],
        isLoading: false,
        error: null,
        
        assets: [
          {
            id: 'btc-usd',
            symbol: 'BTC',
            name: 'Bitcoin',
            category: 'crypto' as const,
            price: 43250.50,
            change24h: 2.45,
            volume: 28500000000,
            marketCap: 847000000000,
            lastUpdated: new Date().toISOString()
          },
          {
            id: 'eth-usd',
            symbol: 'ETH',
            name: 'Ethereum',
            category: 'crypto' as const,
            price: 2650.75,
            change24h: -1.23,
            volume: 15200000000,
            marketCap: 318000000000,
            lastUpdated: new Date().toISOString()
          },
          {
            id: 'aapl',
            symbol: 'AAPL',
            name: 'Apple Inc.',
            category: 'stocks' as const,
            price: 185.92,
            change24h: 0.87,
            volume: 52000000,
            marketCap: 2900000000000,
            lastUpdated: new Date().toISOString()
          }
        ],
        models: [
          // Crypto Models
          {
            id: 'crypto_s2f',
            name: 'Stock-to-Flow (S2F)',
            category: 'crypto' as const,
            type: 'fundamental' as const,
            description: 'Bitcoin scarcity model based on stock-to-flow ratio',
            accuracy: 85.2,
            lastRun: new Date(Date.now() - 1800000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'crypto_metcalfe',
            name: "Metcalfe's Law",
            category: 'crypto' as const,
            type: 'fundamental' as const,
            description: 'Network value analysis based on active addresses',
            accuracy: 72.8,
            lastRun: new Date(Date.now() - 2400000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'crypto_nvt',
            name: 'NVT / NVM',
            category: 'crypto' as const,
            type: 'fundamental' as const,
            description: 'Network value to transactions ratio analysis',
            accuracy: 68.5,
            lastRun: new Date(Date.now() - 3600000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'crypto_finbert',
            name: 'FinBERT / CryptoBERT',
            category: 'crypto' as const,
            type: 'sentiment' as const,
            description: 'NLP sentiment analysis for crypto markets',
            accuracy: 76.3,
            lastRun: new Date(Date.now() - 1200000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'crypto_rl',
            name: 'Reinforcement Learning',
            category: 'crypto' as const,
            type: 'ml' as const,
            description: 'Adaptive trading strategies using RL algorithms',
            accuracy: 82.1,
            lastRun: new Date(Date.now() - 900000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          
          // Stock Models
          {
            id: 'stock_dcf',
            name: 'DCF Model',
            category: 'stocks' as const,
            type: 'fundamental' as const,
            description: 'Discounted Cash Flow valuation model',
            accuracy: 74.6,
            lastRun: new Date(Date.now() - 7200000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'stock_capm',
            name: 'CAPM',
            category: 'stocks' as const,
            type: 'fundamental' as const,
            description: 'Capital Asset Pricing Model for risk assessment',
            accuracy: 69.8,
            lastRun: new Date(Date.now() - 5400000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'stock_lstm',
            name: 'LSTM Neural Network',
            category: 'stocks' as const,
            type: 'ml' as const,
            description: 'Long Short-Term Memory network for stock prediction',
            accuracy: 78.5,
            lastRun: new Date(Date.now() - 3600000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'stock_xgboost',
            name: 'XGBoost',
            category: 'stocks' as const,
            type: 'ml' as const,
            description: 'Gradient boosting for stock price prediction',
            accuracy: 81.2,
            lastRun: new Date(Date.now() - 2700000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          
          // Forex Models
          {
            id: 'forex_ppp',
            name: 'Purchasing Power Parity',
            category: 'forex' as const,
            type: 'fundamental' as const,
            description: 'PPP model for currency valuation',
            accuracy: 65.4,
            lastRun: new Date(Date.now() - 4800000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'forex_lstm',
            name: 'LSTM for Forex',
            category: 'forex' as const,
            type: 'ml' as const,
            description: 'Deep learning model for currency pair prediction',
            accuracy: 73.7,
            lastRun: new Date(Date.now() - 1800000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          
          // Cross-Asset Models
          {
            id: 'cross_rsi',
            name: 'RSI Momentum',
            category: 'cross-asset' as const,
            type: 'technical' as const,
            description: 'Relative Strength Index based momentum analysis',
            accuracy: 65.2,
            lastRun: new Date(Date.now() - 7200000).toISOString(),
            isActive: true,
            status: 'completed' as const
          },
          {
            id: 'sentiment-analyzer',
            name: 'News Sentiment Analyzer',
            category: 'cross-asset' as const,
            type: 'sentiment' as const,
            description: 'AI-powered sentiment analysis from news and social media',
            accuracy: 72.8,
            lastRun: new Date(Date.now() - 1800000).toISOString(),
            isActive: true,
            status: 'running' as const
          },
          {
            id: 'cross_arima',
            name: 'ARIMA',
            category: 'cross-asset' as const,
            type: 'statistical' as const,
            description: 'Autoregressive Integrated Moving Average model',
            accuracy: 71.4,
            lastRun: new Date(Date.now() - 5400000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'cross_garch',
            name: 'GARCH',
            category: 'cross-asset' as const,
            type: 'statistical' as const,
            description: 'Generalized Autoregressive Conditional Heteroskedasticity',
            accuracy: 69.8,
            lastRun: new Date(Date.now() - 4200000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'cross_transformer',
            name: 'Transformer',
            category: 'cross-asset' as const,
            type: 'ml' as const,
            description: 'Attention-based neural network for time series',
            accuracy: 84.7,
            lastRun: new Date(Date.now() - 1800000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'cross_lightgbm',
            name: 'LightGBM',
            category: 'cross-asset' as const,
            type: 'ml' as const,
            description: 'Gradient boosting framework for machine learning',
            accuracy: 79.3,
            lastRun: new Date(Date.now() - 2700000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'cross_macd',
            name: 'MACD',
            category: 'cross-asset' as const,
            type: 'technical' as const,
            description: 'Moving Average Convergence Divergence indicator',
            accuracy: 62.8,
            lastRun: new Date(Date.now() - 6300000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'cross_ichimoku',
            name: 'Ichimoku',
            category: 'cross-asset' as const,
            type: 'technical' as const,
            description: 'Ichimoku Kinko Hyo comprehensive indicator',
            accuracy: 67.5,
            lastRun: new Date(Date.now() - 3900000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
          {
            id: 'cross_ppo',
            name: 'PPO (RL)',
            category: 'cross-asset' as const,
            type: 'ml' as const,
            description: 'Proximal Policy Optimization reinforcement learning',
            accuracy: 80.6,
            lastRun: new Date(Date.now() - 2100000).toISOString(),
            isActive: true,
            status: 'idle' as const
          },
           {
             id: 'cross_markowitz',
             name: 'Markowitz MPT',
             category: 'cross-asset' as const,
             type: 'fundamental' as const,
             description: 'Modern Portfolio Theory optimization',
             accuracy: 73.2,
             lastRun: new Date(Date.now() - 4800000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           
           // Futures Models
           {
             id: 'futures_carry',
             name: 'Cost-of-Carry Model',
             category: 'futures' as const,
             type: 'fundamental' as const,
             description: 'Futures pricing based on cost of carry',
             accuracy: 76.8,
             lastRun: new Date(Date.now() - 3600000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'futures_convenience',
             name: 'Convenience Yield',
             category: 'futures' as const,
             type: 'fundamental' as const,
             description: 'Commodity futures convenience yield model',
             accuracy: 68.4,
             lastRun: new Date(Date.now() - 5400000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'futures_samuelson',
             name: 'Samuelson Effect',
             category: 'futures' as const,
             type: 'fundamental' as const,
             description: 'Volatility increase as expiration approaches',
             accuracy: 71.6,
             lastRun: new Date(Date.now() - 4200000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'futures_rl',
             name: 'RL (SAC)',
             category: 'futures' as const,
             type: 'ml' as const,
             description: 'Soft Actor-Critic reinforcement learning',
             accuracy: 78.9,
             lastRun: new Date(Date.now() - 1800000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           
           // Index Models
           {
             id: 'index_apt',
             name: 'Arbitrage Pricing Theory',
             category: 'indexes' as const,
             type: 'fundamental' as const,
             description: 'Multi-factor asset pricing model',
             accuracy: 72.3,
             lastRun: new Date(Date.now() - 6300000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'index_ddm',
             name: 'Dividend Discount Model',
             category: 'indexes' as const,
             type: 'fundamental' as const,
             description: 'Index valuation based on dividend streams',
             accuracy: 69.7,
             lastRun: new Date(Date.now() - 7200000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'index_kalman',
             name: 'Kalman Filters',
             category: 'indexes' as const,
             type: 'statistical' as const,
             description: 'State-space model for dynamic estimation',
             accuracy: 75.8,
             lastRun: new Date(Date.now() - 3600000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'index_vecm',
             name: 'VECM',
             category: 'indexes' as const,
             type: 'statistical' as const,
             description: 'Vector Error Correction Model for cointegration',
             accuracy: 73.5,
             lastRun: new Date(Date.now() - 4800000).toISOString(),
             isActive: true,
             status: 'idle' as const
           },
           {
             id: 'index_elliott',
             name: 'Elliott Wave',
             category: 'indexes' as const,
             type: 'technical' as const,
             description: 'Elliott Wave pattern recognition',
             accuracy: 58.9,
             lastRun: new Date(Date.now() - 8100000).toISOString(),
             isActive: false,
             status: 'idle' as const
           }
         ],
        news: [],
        forumPosts: [],
        user: null,
        chartData: {},
        modelPredictions: [],
        aiInsights: [],
        watchlist: [
          {
            id: 'btc-usd',
            symbol: 'BTC',
            name: 'Bitcoin',
            category: 'crypto' as const,
            price: 43250.50,
            change24h: 2.45,
            volume: 28500000000,
            marketCap: 847000000000,
            lastUpdated: new Date().toISOString(),
            isFavorite: true
          },
          {
            id: 'eth-usd',
            symbol: 'ETH',
            name: 'Ethereum',
            category: 'crypto' as const,
            price: 2650.75,
            change24h: -1.23,
            volume: 15200000000,
            marketCap: 318000000000,
            lastUpdated: new Date().toISOString(),
            isFavorite: false
          },
          {
            id: 'aapl',
            symbol: 'AAPL',
            name: 'Apple Inc.',
            category: 'stocks' as const,
            price: 185.92,
            change24h: 0.87,
            volume: 52000000,
            marketCap: 2900000000000,
            lastUpdated: new Date().toISOString(),
            isFavorite: true
          }
        ],
        notifications: [],
        
        socket: null,
        isConnected: false,
        
        searchQuery: '',
        newsFilters: {
          category: 'all',
          sentiment: 'all',
          dateRange: '24h'
        },
        assetFilters: {
          category: 'all',
          sortBy: 'marketCap',
          sortOrder: 'desc'
        },
        
        // UI Actions
        setActiveView: (view) => set({ activeView: view }),
        
        addSelectedAsset: (asset) => {
          const { selectedAssets } = get();
          if (!selectedAssets.find(a => a.id === asset.id) && selectedAssets.length < 10) {
            set({ selectedAssets: [...selectedAssets, asset] });
          }
        },
        
        removeSelectedAsset: (assetId) => {
          const { selectedAssets } = get();
          set({ selectedAssets: selectedAssets.filter(a => a.id !== assetId) });
        },
        
        addSelectedModel: (model) => {
          const { selectedModels } = get();
          if (!selectedModels.find(m => m.id === model.id) && selectedModels.length < 5) {
            set({ selectedModels: [...selectedModels, { ...model, isActive: true }] });
          }
        },
        
        removeSelectedModel: (modelId) => {
          const { selectedModels } = get();
          set({ selectedModels: selectedModels.filter(m => m.id !== modelId) });
        },
        
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),
        
        // Data Actions
        updateAssets: (assets) => set({ assets }),
        setAssets: (assets) => set({ assets }),
        updateAsset: (asset) => {
          const { assets } = get();
          const updatedAssets = assets.map(a => a.id === asset.id ? asset : a);
          set({ assets: updatedAssets });
        },
        updateModels: (models) => set({ models }),
        updateNews: (news) => set({ news }),
        updateForumPosts: (posts) => set({ forumPosts: posts }),
        setUser: (user) => set({ user }),
        updateChartData: (assetId, data) => {
          const { chartData } = get();
          set({ chartData: { ...chartData, [assetId]: data } });
        },
        setChartData: (assetId, data) => {
          const { chartData } = get();
          set({ chartData: { ...chartData, [assetId]: data } });
        },
        addModelPrediction: (prediction) => {
          const { modelPredictions } = get();
          set({ modelPredictions: [prediction, ...modelPredictions.slice(0, 99)] });
        },
        addAIInsight: (insight) => {
          const { aiInsights } = get();
          set({ aiInsights: [...(aiInsights || []), insight] });
        },
        
        // Watchlist Actions
        addToWatchlist: (asset) => {
          const { watchlist } = get();
          if (!watchlist.find(a => a.id === asset.id)) {
            set({ watchlist: [...watchlist, asset] });
          }
        },
        
        removeFromWatchlist: (assetId) => {
          const { watchlist } = get();
          set({ watchlist: watchlist.filter(a => a.id !== assetId) });
        },
        
        // Notification Actions
        addNotification: (notification) => {
          const { notifications } = get();
          set({ notifications: [notification, ...notifications] });
        },
        
        markNotificationAsRead: (notificationId) => {
          const { notifications } = get();
          const updatedNotifications = notifications.map(n => 
            n.id === notificationId ? { ...n, read: true } : n
          );
          set({ notifications: updatedNotifications });
        },
        
        clearNotifications: () => {
          set({ notifications: [] });
        },
        
        // WebSocket Actions
        connectWebSocket: () => {
          console.log('🔌 Attempting to connect WebSocket to:', WS_URL);
          const socket = io(WS_URL, {
            transports: ['websocket'],
            autoConnect: true
          });
          
          socket.on('connect_error', (error) => {
            console.error('❌ WebSocket connection error:', error);
            set({ isConnected: false, error: error.message });
          });
          
          socket.on('connect', () => {
            set({ isConnected: true });
            console.log('WebSocket connected');
          });
          
          socket.on('disconnect', () => {
            set({ isConnected: false });
            console.log('WebSocket disconnected');
          });
          
          socket.on('asset_update', (asset: Asset) => {
            get().updateAsset(asset);
          });
          
          socket.on('news_update', (newsItem: NewsItem) => {
            const { news } = get();
            set({ news: [newsItem, ...news.slice(0, 99)] });
          });
          
          socket.on('model_prediction', (prediction: ModelPrediction) => {
            get().addModelPrediction(prediction);
          });
          
          set({ socket });
        },
        
        disconnectWebSocket: () => {
          const { socket } = get();
          if (socket) {
            socket.disconnect();
            set({ socket: null, isConnected: false });
          }
        },
        
        // Search & Filter Actions
        setSearchQuery: (query) => set({ searchQuery: query }),
        setNewsFilters: (filters) => {
          const { newsFilters } = get();
          set({ newsFilters: { ...newsFilters, ...filters } });
        },
        setAssetFilters: (filters) => {
          const { assetFilters } = get();
          set({ assetFilters: { ...assetFilters, ...filters } });
        },
        
        // API Actions
        fetchAssets: async () => {
          try {
            set({ isLoading: true, error: null });
            const response = await fetch(`${API_BASE_URL}/market/assets`);
            if (!response.ok) throw new Error('Failed to fetch assets');
            const assets = await response.json();
            set({ assets, isLoading: false });
          } catch (error) {
            set({ error: (error as Error).message, isLoading: false });
          }
        },
        
        fetchModels: async () => {
          try {
            set({ isLoading: true, error: null });
            // Mock models data since backend doesn't have this endpoint yet
            const models = [
              { id: 1, name: 'LSTM Price Predictor', type: 'neural_network', accuracy: 0.85 },
              { id: 2, name: 'Random Forest Classifier', type: 'ensemble', accuracy: 0.78 },
              { id: 3, name: 'Sentiment Analysis', type: 'nlp', accuracy: 0.92 }
            ];
            set({ models, isLoading: false });
          } catch (error) {
            set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
          }
        },
        
        fetchNews: async () => {
          try {
            set({ isLoading: true, error: null });
            const response = await fetch(`${API_BASE_URL}/news`);
            if (!response.ok) throw new Error('Failed to fetch news');
            const news = await response.json();
            set({ news, isLoading: false });
          } catch (error) {
            set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
          }
        },
        
        fetchForumPosts: async () => {
          try {
            set({ isLoading: true, error: null });
            // Mock forum posts data since backend doesn't have this endpoint yet
            const posts = [
              { id: 1, title: 'Bitcoin Analysis Discussion', author: 'CryptoTrader', replies: 15, likes: 23 },
              { id: 2, title: 'Stock Market Predictions', author: 'MarketGuru', replies: 8, likes: 12 },
              { id: 3, title: 'AI Trading Strategies', author: 'AIExpert', replies: 22, likes: 45 }
            ];
            set({ forumPosts: posts, isLoading: false });
          } catch (error) {
            set({ error: error instanceof Error ? error.message : 'Unknown error', isLoading: false });
          }
        },
        
        fetchChartData: async (assetId: string, timeframe: string = '1d') => {
          try {
            set({ isLoading: true, error: null });
            // Mock chart data since backend doesn't have this endpoint yet
            const mockData = [];
            const basePrice = Math.random() * 1000 + 100;
            for (let i = 0; i < 100; i++) {
              const timestamp = Date.now() - (100 - i) * 60000;
              const price = basePrice + (Math.random() - 0.5) * 20;
              mockData.push({
                timestamp,
                open: price,
                high: price + Math.random() * 5,
                low: price - Math.random() * 5,
                close: price + (Math.random() - 0.5) * 2,
                volume: Math.random() * 1000000
              });
            }
            get().updateChartData(assetId, mockData);
            set({ isLoading: false });
          } catch (error) {
            set({ error: (error as Error).message, isLoading: false });
          }
        },

        runModel: async (modelId: string, assetId: string) => {
          try {
            set({ isLoading: true, error: null });
            // Mock model prediction since backend doesn't have this endpoint yet
            const prediction = {
              modelId,
              assetId,
              prediction: {
                direction: Math.random() > 0.5 ? 'bullish' : 'bearish' as 'bullish' | 'bearish',
                confidence: Math.random() * 0.4 + 0.6,
                targetPrice: Math.random() * 1000 + 100,
                timeframe: '1d' as '1h' | '4h' | '1d' | '1w' | '1m',
                reasoning: 'Mock prediction based on technical analysis'
              },
              timestamp: new Date().toISOString()
            };
            get().addModelPrediction(prediction);
            set({ isLoading: false });
          } catch (error) {
            set({ error: (error as Error).message, isLoading: false });
          }
        },

        stopModel: async (modelId: string) => {
          try {
            set({ isLoading: true, error: null });
            // Mock model stop since backend doesn't have this endpoint yet
            const { models } = get();
            const updatedModels = models.map(model => 
              model.id === modelId ? { ...model, status: 'idle' as const } : model
            );
            set({ models: updatedModels, isLoading: false });
          } catch (error) {
            set({ error: (error as Error).message, isLoading: false });
          }
        }
      }),
      {
        name: 'finscope-store',
        partialize: (state) => ({
          selectedAssets: state.selectedAssets,
          selectedModels: state.selectedModels,
          user: state.user,
          newsFilters: state.newsFilters,
          assetFilters: state.assetFilters
        })
      }
    ),
    { name: 'FinScope Store' }
  ));

// Selectors
export const useSelectedAssets = () => useStore(state => state.selectedAssets);
export const useSelectedModels = () => useStore(state => state.selectedModels);
export const useActiveView = () => useStore(state => state.activeView);
export const useIsLoading = () => useStore(state => state.isLoading);
export const useError = () => useStore(state => state.error);
export const useAssets = () => useStore(state => state.assets);
export const useModels = () => useStore(state => state.models);
export const useNews = () => useStore(state => state.news);
export const useForumPosts = () => useStore(state => state.forumPosts);
export const useUser = () => useStore(state => state.user);
export const useWebSocketStatus = () => useStore(state => ({ isConnected: state.isConnected, socket: state.socket }));
export const useModelPredictions = () => useStore(state => state.modelPredictions);

// Filtered selectors
export const useFilteredNews = () => {
  const news = useStore(state => state.news);
  const filters = useStore(state => state.newsFilters);
  const searchQuery = useStore(state => state.searchQuery);
  
  return news.filter(item => {
    const matchesCategory = filters.category === 'all' || item.category === filters.category;
    const matchesSentiment = filters.sentiment === 'all' || item.sentiment === filters.sentiment;
    const matchesSearch = !searchQuery || 
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.summary.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesCategory && matchesSentiment && matchesSearch;
  });
};

export const useFilteredAssets = () => {
  const assets = useStore(state => state.assets);
  const filters = useStore(state => state.assetFilters);
  const searchQuery = useStore(state => state.searchQuery);
  
  return assets
    .filter(asset => {
      const matchesCategory = filters.category === 'all' || asset.category === filters.category;
      const matchesSearch = !searchQuery || 
        asset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        asset.symbol.toLowerCase().includes(searchQuery.toLowerCase());
      
      return matchesCategory && matchesSearch;
    })
    .sort((a, b) => {
      const { sortBy, sortOrder } = filters;
      const aValue = a[sortBy as keyof Asset] as number;
      const bValue = b[sortBy as keyof Asset] as number;
      
      if (sortOrder === 'asc') {
        return aValue - bValue;
      } else {
        return bValue - aValue;
      }
    });
};