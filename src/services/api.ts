import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { Asset, Model, NewsItem, ForumPost, User, ChartData, ModelPrediction } from '../store/useStore';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_TIMEOUT = 30000; // 30 seconds

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('finscope_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('finscope_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API Response Types
interface ApiResponse<T> {
  data: T;
  message?: string;
  status: string;
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// Authentication API
export const authAPI = {
  login: async (email: string, password: string): Promise<{ user: User; token: string }> => {
    const response = await apiClient.post<ApiResponse<{ user: User; token: string }>>('/auth/login', {
      email,
      password,
    });
    return response.data.data;
  },

  register: async (userData: {
    username: string;
    email: string;
    password: string;
  }): Promise<{ user: User; token: string }> => {
    const response = await apiClient.post<ApiResponse<{ user: User; token: string }>>('/auth/register', userData);
    return response.data.data;
  },

  logout: async (): Promise<void> => {
    await apiClient.post('/auth/logout');
    localStorage.removeItem('finscope_token');
  },

  refreshToken: async (): Promise<{ token: string }> => {
    const response = await apiClient.post<ApiResponse<{ token: string }>>('/auth/refresh');
    return response.data.data;
  },

  getProfile: async (): Promise<User> => {
    const response = await apiClient.get<ApiResponse<User>>('/auth/profile');
    return response.data.data;
  },

  updateProfile: async (userData: Partial<User>): Promise<User> => {
    const response = await apiClient.put<ApiResponse<User>>('/auth/profile', userData);
    return response.data.data;
  },
};

// Assets API
export const assetsAPI = {
  getAssets: async (params?: {
    category?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<PaginatedResponse<Asset>> => {
    const response = await apiClient.get<Asset[]>('/market/assets', { params });
    // Transform to match expected format
    return {
      data: response.data,
      total: response.data.length,
      page: 1,
      limit: response.data.length,
      hasNext: false,
      hasPrev: false
    };
  },

  getAsset: async (assetId: string): Promise<Asset> => {
    const response = await apiClient.get<Asset>(`/market/asset/${assetId}`);
    return response.data;
  },

  getAssetChart: async (
    assetId: string,
    timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M',
    limit?: number
  ): Promise<ChartData[]> => {
    // Mock chart data for now since backend doesn't have this endpoint yet
    const mockData: ChartData[] = [];
    const basePrice = Math.random() * 1000 + 100;
    const dataPoints = limit || 100;
    
    for (let i = 0; i < dataPoints; i++) {
      const timestamp = Date.now() - (dataPoints - i) * 60000;
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
    
    return mockData;
  },

  getAssetFundamentals: async (assetId: string): Promise<any> => {
    const response = await apiClient.get<ApiResponse<any>>(`/api/assets/${assetId}/fundamentals`);
    return response.data.data;
  },

  searchAssets: async (query: string): Promise<Asset[]> => {
    const response = await apiClient.get<ApiResponse<Asset[]>>('/api/assets/search', {
      params: { q: query },
    });
    return response.data.data;
  },
};

// Models API
export const modelsAPI = {
  getModels: async (params?: {
    category?: string;
    type?: string;
    active?: boolean;
  }): Promise<Model[]> => {
    const response = await apiClient.get<ApiResponse<Model[]>>('/api/models', { params });
    return response.data.data;
  },

  getModel: async (modelId: string): Promise<Model> => {
    const response = await apiClient.get<ApiResponse<Model>>(`/api/models/${modelId}`);
    return response.data.data;
  },

  runModel: async (modelId: string, assetId: string, params?: any): Promise<ModelPrediction> => {
    const response = await apiClient.post<ApiResponse<ModelPrediction>>(
      `/api/models/${modelId}/run`,
      { asset_id: assetId, parameters: params }
    );
    return response.data.data;
  },

  getModelPredictions: async (
    modelId?: string,
    assetId?: string,
    limit?: number
  ): Promise<ModelPrediction[]> => {
    const response = await apiClient.get<ApiResponse<ModelPrediction[]>>('/api/models/predictions', {
      params: { model_id: modelId, asset_id: assetId, limit },
    });
    return response.data.data;
  },

  getModelPerformance: async (modelId: string): Promise<any> => {
    const response = await apiClient.get<ApiResponse<any>>(`/api/models/${modelId}/performance`);
    return response.data.data;
  },
};

// News API
export const newsAPI = {
  getNews: async (params?: {
    category?: string;
    sentiment?: string;
    search?: string;
    limit?: number;
    offset?: number;
    date_from?: string;
    date_to?: string;
  }): Promise<PaginatedResponse<NewsItem>> => {
    const response = await apiClient.get<NewsItem[]>('/news', { params });
    // Transform to match expected format
    return {
      data: response.data,
      total: response.data.length,
      page: 1,
      limit: response.data.length,
      hasNext: false,
      hasPrev: false
    };
  },

  getNewsItem: async (newsId: string): Promise<NewsItem> => {
    const response = await apiClient.get<ApiResponse<NewsItem>>(`/api/news/${newsId}`);
    return response.data.data;
  },

  getNewsSentiment: async (text: string): Promise<{ sentiment: string; confidence: number }> => {
    const response = await apiClient.post<ApiResponse<{ sentiment: string; confidence: number }>>(
      '/api/news/sentiment',
      { text }
    );
    return response.data.data;
  },

  getNewsForAsset: async (assetId: string, limit?: number): Promise<NewsItem[]> => {
    const response = await apiClient.get<ApiResponse<NewsItem[]>>(
      `/api/news/asset/${assetId}`,
      { params: { limit } }
    );
    return response.data.data;
  },
};

// Forum API
export const forumAPI = {
  getPosts: async (params?: {
    category?: string;
    search?: string;
    sort?: 'newest' | 'oldest' | 'popular' | 'trending';
    limit?: number;
    offset?: number;
  }): Promise<PaginatedResponse<ForumPost>> => {
    const response = await apiClient.get<PaginatedResponse<ForumPost>>('/api/forum/posts', { params });
    return response.data;
  },

  getPost: async (postId: string): Promise<ForumPost> => {
    const response = await apiClient.get<ApiResponse<ForumPost>>(`/api/forum/posts/${postId}`);
    return response.data.data;
  },

  createPost: async (postData: {
    title: string;
    content: string;
    category: string;
    tags: string[];
  }): Promise<ForumPost> => {
    const response = await apiClient.post<ApiResponse<ForumPost>>('/api/forum/posts', postData);
    return response.data.data;
  },

  updatePost: async (postId: string, postData: Partial<ForumPost>): Promise<ForumPost> => {
    const response = await apiClient.put<ApiResponse<ForumPost>>(`/api/forum/posts/${postId}`, postData);
    return response.data.data;
  },

  deletePost: async (postId: string): Promise<void> => {
    await apiClient.delete(`/api/forum/posts/${postId}`);
  },

  votePost: async (postId: string, vote: 'up' | 'down'): Promise<{ upvotes: number; downvotes: number }> => {
    const response = await apiClient.post<ApiResponse<{ upvotes: number; downvotes: number }>>(
      `/api/forum/posts/${postId}/vote`,
      { vote }
    );
    return response.data.data;
  },

  getComments: async (postId: string): Promise<any[]> => {
    const response = await apiClient.get<ApiResponse<any[]>>(`/api/forum/posts/${postId}/comments`);
    return response.data.data;
  },

  createComment: async (postId: string, content: string): Promise<any> => {
    const response = await apiClient.post<ApiResponse<any>>(
      `/api/forum/posts/${postId}/comments`,
      { content }
    );
    return response.data.data;
  },
};

// Watchlist API
export const watchlistAPI = {
  getWatchlists: async (): Promise<any[]> => {
    const response = await apiClient.get<ApiResponse<any[]>>('/api/watchlists');
    return response.data.data;
  },

  createWatchlist: async (name: string, description?: string): Promise<any> => {
    const response = await apiClient.post<ApiResponse<any>>('/api/watchlists', {
      name,
      description,
    });
    return response.data.data;
  },

  addToWatchlist: async (watchlistId: string, assetId: string): Promise<void> => {
    await apiClient.post(`/api/watchlists/${watchlistId}/assets`, { asset_id: assetId });
  },

  removeFromWatchlist: async (watchlistId: string, assetId: string): Promise<void> => {
    await apiClient.delete(`/api/watchlists/${watchlistId}/assets/${assetId}`);
  },

  deleteWatchlist: async (watchlistId: string): Promise<void> => {
    await apiClient.delete(`/api/watchlists/${watchlistId}`);
  },
};

// Portfolio API
export const portfolioAPI = {
  getPortfolio: async (): Promise<any> => {
    const response = await apiClient.get<ApiResponse<any>>('/api/portfolio');
    return response.data.data;
  },

  addPosition: async (positionData: {
    asset_id: string;
    quantity: number;
    entry_price: number;
    entry_date: string;
  }): Promise<any> => {
    const response = await apiClient.post<ApiResponse<any>>('/api/portfolio/positions', positionData);
    return response.data.data;
  },

  updatePosition: async (positionId: string, positionData: any): Promise<any> => {
    const response = await apiClient.put<ApiResponse<any>>(
      `/api/portfolio/positions/${positionId}`,
      positionData
    );
    return response.data.data;
  },

  deletePosition: async (positionId: string): Promise<void> => {
    await apiClient.delete(`/api/portfolio/positions/${positionId}`);
  },

  getPortfolioAnalytics: async (): Promise<any> => {
    const response = await apiClient.get<ApiResponse<any>>('/api/portfolio/analytics');
    return response.data.data;
  },
};

// AI/LLM API
export const aiAPI = {
  getExplanation: async (data: {
    asset_id: string;
    model_predictions?: ModelPrediction[];
    news_context?: NewsItem[];
    question?: string;
  }): Promise<{ explanation: string; confidence: number }> => {
    const response = await apiClient.get<{ explanation: string; confidence: number }>(
      `/ai/analyze/${data.asset_id}`
    );
    return response.data;
  },

  askQuestion: async (question: string, context?: any): Promise<{ answer: string; sources: any[] }> => {
    const response = await apiClient.post<ApiResponse<{ answer: string; sources: any[] }>>(
      '/api/ai/ask',
      { question, context }
    );
    return response.data.data;
  },

  generateTradingPlan: async (data: {
    asset_id: string;
    risk_tolerance: 'low' | 'medium' | 'high';
    investment_horizon: 'short' | 'medium' | 'long';
    capital: number;
  }): Promise<any> => {
    const response = await apiClient.post<ApiResponse<any>>('/api/ai/trading-plan', data);
    return response.data.data;
  },

  getMarketSummary: async (): Promise<{ summary: string; key_events: any[] }> => {
    const response = await apiClient.get<ApiResponse<{ summary: string; key_events: any[] }>>(
      '/api/ai/market-summary'
    );
    return response.data.data;
  },
};

// Search API
export const searchAPI = {
  globalSearch: async (query: string): Promise<{
    assets: Asset[];
    news: NewsItem[];
    posts: ForumPost[];
    models: Model[];
  }> => {
    const response = await apiClient.get<ApiResponse<{
      assets: Asset[];
      news: NewsItem[];
      posts: ForumPost[];
      models: Model[];
    }>>('/api/search', { params: { q: query } });
    return response.data.data;
  },

  semanticSearch: async (query: string, type?: 'assets' | 'news' | 'posts'): Promise<any[]> => {
    const response = await apiClient.get<ApiResponse<any[]>>('/api/search/semantic', {
      params: { q: query, type },
    });
    return response.data.data;
  },
};

// Notifications API
export const notificationsAPI = {
  getNotifications: async (limit?: number): Promise<any[]> => {
    const response = await apiClient.get<ApiResponse<any[]>>('/api/notifications', {
      params: { limit },
    });
    return response.data.data;
  },

  markAsRead: async (notificationId: string): Promise<void> => {
    await apiClient.put(`/api/notifications/${notificationId}/read`);
  },

  markAllAsRead: async (): Promise<void> => {
    await apiClient.put('/api/notifications/read-all');
  },

  deleteNotification: async (notificationId: string): Promise<void> => {
    await apiClient.delete(`/api/notifications/${notificationId}`);
  },

  updateSettings: async (settings: any): Promise<any> => {
    const response = await apiClient.put<ApiResponse<any>>('/api/notifications/settings', settings);
    return response.data.data;
  },
};

// Export the main API client for custom requests
export { apiClient };

// Utility functions
export const setAuthToken = (token: string): void => {
  localStorage.setItem('finscope_token', token);
};

export const getAuthToken = (): string | null => {
  return localStorage.getItem('finscope_token');
};

export const clearAuthToken = (): void => {
  localStorage.removeItem('finscope_token');
};

// Error handling utility
export const handleApiError = (error: any): string => {
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  if (error.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
};