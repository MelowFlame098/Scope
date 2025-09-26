"use client";

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Types
export interface Asset {
  id: string;
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  sector: string;
  lastUpdated: string;
}

export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  content: string;
  source: string;
  author: string;
  publishedAt: string;
  url: string;
  imageUrl?: string;
  category: string;
  tags: string[];
  sentiment: 'positive' | 'negative' | 'neutral';
  relevanceScore: number;
}

export interface Model {
  id: string;
  name: string;
  description: string;
  type: 'prediction' | 'analysis' | 'sentiment';
  accuracy: number;
  status: 'active' | 'inactive' | 'training';
  lastTrained: string;
  parameters: Record<string, any>;
}

export interface ChartData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ModelPrediction {
  id: string;
  modelId: string;
  assetId: string;
  prediction: {
    direction: 'up' | 'down' | 'sideways';
    confidence: number;
    targetPrice: number;
    timeframe: string;
  };
  createdAt: string;
}

export interface User {
  id: string;
  email: string;
  username: string;
  fullName: string;
  avatar?: string;
  plan: 'free' | 'basic' | 'premium';
  isEmailVerified: boolean;
  preferences: {
    theme: 'light' | 'dark';
    notifications: boolean;
    defaultTimeframe: string;
  };
}

export interface Notification {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  read: boolean;
  timestamp: string;
}

// Store State Interface
interface StoreState {
  // Data
  assets: Asset[];
  news: NewsItem[];
  models: Model[];
  chartData: Record<string, ChartData[]>;
  modelPredictions: ModelPrediction[];
  notifications: Notification[];
  user: User | null;
  watchlist: Asset[];
  
  // UI State
  selectedAssets: string[];
  selectedModels: string[];
  isLoading: boolean;
  error: string | null;
  isConnected: boolean;
  
  // Search & Filters
  searchQuery: string;
  newsFilters: {
    category: string;
    sentiment: string;
    source: string;
    dateRange: string;
  };
  assetFilters: {
    sector: string;
    priceRange: [number, number];
    marketCap: string;
  };
}

// Store Actions Interface
interface StoreActions {
  // Basic Actions
  setAssets: (assets: Asset[]) => void;
  updateAsset: (asset: Asset) => void;
  updateModels: (models: Model[]) => void;
  updateNews: (news: NewsItem[]) => void;
  setUser: (user: User | null) => void;
  updateChartData: (assetId: string, data: ChartData[]) => void;
  setChartData: (assetId: string, data: ChartData[]) => void;
  addModelPrediction: (prediction: ModelPrediction) => void;
  
  // Watchlist Actions
  addToWatchlist: (asset: Asset) => void;
  removeFromWatchlist: (assetId: string) => void;
  
  // Notification Actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  markNotificationAsRead: (notificationId: string) => void;
  clearNotifications: () => void;
  
  // Search & Filter Actions
  setSearchQuery: (query: string) => void;
  setNewsFilters: (filters: Partial<StoreState['newsFilters']>) => void;
  setAssetFilters: (filters: Partial<StoreState['assetFilters']>) => void;
  
  // Selection Actions
  setSelectedAssets: (assetIds: string[]) => void;
  setSelectedModels: (modelIds: string[]) => void;
  
  // Loading & Error Actions
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setConnected: (connected: boolean) => void;
  
  // API Actions
  fetchAssets: () => Promise<void>;
  fetchModels: () => Promise<void>;
  fetchNews: () => Promise<void>;
  fetchChartData: (assetId: string, timeframe: string) => Promise<void>;
  runModel: (modelId: string, assetId: string) => Promise<void>;
  stopModel: (modelId: string) => Promise<void>;
}

type Store = StoreState & StoreActions;

// Initial State
const initialState: StoreState = {
  assets: [],
  news: [],
  models: [],
  chartData: {},
  modelPredictions: [],
  notifications: [],
  user: null,
  watchlist: [],
  selectedAssets: [],
  selectedModels: [],
  isLoading: false,
  error: null,
  isConnected: false,
  searchQuery: '',
  newsFilters: {
    category: 'all',
    sentiment: 'all',
    source: 'all',
    dateRange: '24h'
  },
  assetFilters: {
    sector: 'all',
    priceRange: [0, 1000000],
    marketCap: 'all'
  }
};

// Mock API functions
const mockFetchAssets = async (): Promise<Asset[]> => {
  await new Promise(resolve => setTimeout(resolve, 1000));
  return [
    {
      id: '1',
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 175.43,
      change: 2.34,
      changePercent: 1.35,
      volume: 45234567,
      marketCap: 2800000000000,
      sector: 'Technology',
      lastUpdated: new Date().toISOString()
    },
    {
      id: '2',
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      price: 142.56,
      change: -1.23,
      changePercent: -0.85,
      volume: 23456789,
      marketCap: 1800000000000,
      sector: 'Technology',
      lastUpdated: new Date().toISOString()
    },
    {
      id: '3',
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      price: 378.92,
      change: 4.56,
      changePercent: 1.22,
      volume: 34567890,
      marketCap: 2900000000000,
      sector: 'Technology',
      lastUpdated: new Date().toISOString()
    }
  ];
};

const mockFetchModels = async (): Promise<Model[]> => {
  await new Promise(resolve => setTimeout(resolve, 800));
  return [
    {
      id: '1',
      name: 'LSTM Price Predictor',
      description: 'Long Short-Term Memory neural network for price prediction',
      type: 'prediction',
      accuracy: 0.78,
      status: 'active',
      lastTrained: new Date().toISOString(),
      parameters: { lookback: 60, epochs: 100 }
    },
    {
      id: '2',
      name: 'Sentiment Analyzer',
      description: 'Natural language processing model for news sentiment analysis',
      type: 'sentiment',
      accuracy: 0.85,
      status: 'active',
      lastTrained: new Date().toISOString(),
      parameters: { model: 'bert-base', threshold: 0.7 }
    }
  ];
};

const mockFetchNews = async (): Promise<NewsItem[]> => {
  await new Promise(resolve => setTimeout(resolve, 600));
  return [
    {
      id: '1',
      title: 'Federal Reserve Signals Potential Rate Cut',
      summary: 'Fed officials hint at possible monetary policy adjustments amid cooling inflation.',
      content: 'Full article content...',
      source: 'Reuters',
      author: 'John Doe',
      publishedAt: new Date().toISOString(),
      url: 'https://example.com/news/1',
      category: 'monetary-policy',
      tags: ['fed', 'rates', 'inflation'],
      sentiment: 'positive',
      relevanceScore: 0.95
    },
    {
      id: '2',
      title: 'Tech Stocks Rally on Strong Earnings',
      summary: 'Major technology companies exceed expectations in Q4 earnings reports.',
      content: 'Full article content...',
      source: 'Bloomberg',
      author: 'Jane Smith',
      publishedAt: new Date().toISOString(),
      url: 'https://example.com/news/2',
      category: 'earnings',
      tags: ['tech', 'earnings', 'stocks'],
      sentiment: 'positive',
      relevanceScore: 0.88
    }
  ];
};

// Create Store
export const useStore = create<Store>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,
        
        // Basic Actions
        setAssets: (assets) => set({ assets }),
        updateAsset: (asset) => {
          const { assets } = get();
          const updatedAssets = assets.map(a => a.id === asset.id ? asset : a);
          set({ assets: updatedAssets });
        },
        updateModels: (models) => set({ models }),
        updateNews: (news) => set({ news }),
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
          const newNotification: Notification = {
            ...notification,
            id: Date.now().toString(),
            timestamp: new Date().toISOString()
          };
          set({ notifications: [newNotification, ...notifications] });
        },
        markNotificationAsRead: (notificationId) => {
          const { notifications } = get();
          const updatedNotifications = notifications.map(n => 
            n.id === notificationId ? { ...n, read: true } : n
          );
          set({ notifications: updatedNotifications });
        },
        clearNotifications: () => set({ notifications: [] }),
        
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
        
        // Selection Actions
        setSelectedAssets: (assetIds) => set({ selectedAssets: assetIds }),
        setSelectedModels: (modelIds) => set({ selectedModels: modelIds }),
        
        // Loading & Error Actions
        setLoading: (loading) => set({ isLoading: loading }),
        setError: (error) => set({ error }),
        setConnected: (connected) => set({ isConnected: connected }),
        
        // API Actions
        fetchAssets: async () => {
          set({ isLoading: true, error: null });
          try {
            const assets = await mockFetchAssets();
            set({ assets, isLoading: false });
          } catch (error) {
            set({ error: 'Failed to fetch assets', isLoading: false });
          }
        },
        
        fetchModels: async () => {
          set({ isLoading: true, error: null });
          try {
            const models = await mockFetchModels();
            set({ models, isLoading: false });
          } catch (error) {
            set({ error: 'Failed to fetch models', isLoading: false });
          }
        },
        
        fetchNews: async () => {
          set({ isLoading: true, error: null });
          try {
            const news = await mockFetchNews();
            set({ news, isLoading: false });
          } catch (error) {
            set({ error: 'Failed to fetch news', isLoading: false });
          }
        },
        
        fetchChartData: async (assetId, timeframe) => {
          set({ isLoading: true, error: null });
          try {
            // Mock chart data generation
            const data: ChartData[] = [];
            const now = new Date();
            for (let i = 30; i >= 0; i--) {
              const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
              const basePrice = 100 + Math.random() * 50;
              data.push({
                timestamp: date.toISOString(),
                open: basePrice,
                high: basePrice + Math.random() * 5,
                low: basePrice - Math.random() * 5,
                close: basePrice + (Math.random() - 0.5) * 3,
                volume: Math.floor(Math.random() * 1000000)
              });
            }
            
            const { chartData } = get();
            set({ 
              chartData: { ...chartData, [assetId]: data },
              isLoading: false 
            });
          } catch (error) {
            set({ error: 'Failed to fetch chart data', isLoading: false });
          }
        },
        
        runModel: async (modelId, assetId) => {
          set({ isLoading: true, error: null });
          try {
            // Mock model execution
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            const prediction: ModelPrediction = {
              id: Date.now().toString(),
              modelId,
              assetId,
              prediction: {
                direction: Math.random() > 0.5 ? 'up' : 'down',
                confidence: Math.random() * 0.4 + 0.6, // 60-100%
                targetPrice: 100 + Math.random() * 50,
                timeframe: '1d'
              },
              createdAt: new Date().toISOString()
            };
            
            get().addModelPrediction(prediction);
            set({ isLoading: false });
          } catch (error) {
            set({ error: 'Failed to run model', isLoading: false });
          }
        },
        
        stopModel: async (modelId) => {
          set({ isLoading: true, error: null });
          try {
            // Mock model stopping
            await new Promise(resolve => setTimeout(resolve, 500));
            set({ isLoading: false });
          } catch (error) {
            set({ error: 'Failed to stop model', isLoading: false });
          }
        }
      }),
      {
        name: 'finscope-store',
        partialize: (state) => ({
          user: state.user,
          watchlist: state.watchlist,
          newsFilters: state.newsFilters,
          assetFilters: state.assetFilters
        })
      }
    ),
    {
      name: 'finscope-store'
    }
  )
);

export default useStore;