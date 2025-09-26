// API utility functions with authentication handling

interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: number;
}

interface RequestOptions extends RequestInit {
  requiresAuth?: boolean;
  skipRefresh?: boolean;
}

class ApiClient {
  private baseUrl: string;
  private refreshPromise: Promise<boolean> | null = null;

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }

  private async getAuthHeaders(): Promise<Record<string, string>> {
    const token = localStorage.getItem('access_token');
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
  }

  private async refreshToken(): Promise<boolean> {
    // Prevent multiple simultaneous refresh attempts
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = this.performTokenRefresh();
    const result = await this.refreshPromise;
    this.refreshPromise = null;
    return result;
  }

  private async performTokenRefresh(): Promise<boolean> {
    const refreshToken = localStorage.getItem('refresh_token');
    
    if (!refreshToken) {
      return false;
    }

    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refresh_token: refreshToken,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('access_token', data.access_token);
        return true;
      } else {
        // Refresh token is invalid
        this.clearTokens();
        return false;
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
      this.clearTokens();
      return false;
    }
  }

  private clearTokens(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    // Redirect to login if we're not already on an auth page
    if (!window.location.pathname.startsWith('/auth/') && !window.location.pathname.startsWith('/landing')) {
      window.location.href = '/auth/login';
    }
  }

  async request<T = any>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<ApiResponse<T>> {
    const {
      requiresAuth = true,
      skipRefresh = false,
      ...fetchOptions
    } = options;

    const url = `${this.baseUrl}${endpoint}`;
    const headers = requiresAuth ? await this.getAuthHeaders() : {
      'Content-Type': 'application/json',
      ...((fetchOptions.headers as Record<string, string>) || {}),
    };

    try {
      const response = await fetch(url, {
        ...fetchOptions,
        headers: {
          ...headers,
          ...((fetchOptions.headers as Record<string, string>) || {}),
        },
      });

      // Handle 401 Unauthorized - try to refresh token
      if (response.status === 401 && requiresAuth && !skipRefresh) {
        const refreshed = await this.refreshToken();
        
        if (refreshed) {
          // Retry the request with new token
          return this.request(endpoint, { ...options, skipRefresh: true });
        } else {
          // Refresh failed, redirect to login
          this.clearTokens();
          return {
            error: 'Authentication failed. Please log in again.',
            status: 401,
          };
        }
      }

      let data: T | undefined;
      const contentType = response.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      }

      if (response.ok) {
        return {
          data,
          status: response.status,
        };
      } else {
        return {
          error: (data as any)?.detail || (data as any)?.message || `Request failed with status ${response.status}`,
          status: response.status,
        };
      }
    } catch (error) {
      console.error('API request failed:', error);
      return {
        error: error instanceof Error ? error.message : 'Network error occurred',
        status: 0,
      };
    }
  }

  // Convenience methods
  async get<T = any>(endpoint: string, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  async post<T = any>(endpoint: string, data?: any, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T = any>(endpoint: string, data?: any, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async patch<T = any>(endpoint: string, data?: any, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T = any>(endpoint: string, options?: RequestOptions): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }
}

// Create a singleton instance
export const apiClient = new ApiClient();

// Export convenience functions
export const api = {
  get: <T = any>(endpoint: string, options?: RequestOptions) => apiClient.get<T>(endpoint, options),
  post: <T = any>(endpoint: string, data?: any, options?: RequestOptions) => apiClient.post<T>(endpoint, data, options),
  put: <T = any>(endpoint: string, data?: any, options?: RequestOptions) => apiClient.put<T>(endpoint, data, options),
  patch: <T = any>(endpoint: string, data?: any, options?: RequestOptions) => apiClient.patch<T>(endpoint, data, options),
  delete: <T = any>(endpoint: string, options?: RequestOptions) => apiClient.delete<T>(endpoint, options),
};

// Specific API endpoints
export const authApi = {
  login: (email: string, password: string, captcha?: string, rememberMe?: boolean) =>
    api.post('/api/auth/login', {
      email,
      password,
      captcha_token: captcha,
      remember_me: rememberMe,
    }, { requiresAuth: false }),

  register: (userData: {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
    captcha?: string;
    acceptTerms: boolean;
  }) =>
    api.post('/api/auth/register', {
      email: userData.email,
      password: userData.password,
      first_name: userData.firstName,
      last_name: userData.lastName,
      captcha_token: userData.captcha,
      accept_terms: userData.acceptTerms,
    }, { requiresAuth: false }),

  getCurrentUser: () => api.get('/api/auth/me'),

  refreshToken: (refreshToken: string) =>
    api.post('/api/auth/refresh', { refresh_token: refreshToken }, { requiresAuth: false }),

  forgotPassword: (email: string, captcha?: string) =>
    api.post('/api/auth/password-reset', {
      email,
      captcha_token: captcha,
    }, { requiresAuth: false }),

  resetPassword: (token: string, newPassword: string) =>
    api.post('/api/auth/reset-password-confirm', {
      token,
      new_password: newPassword,
    }, { requiresAuth: false }),

  verifyEmail: (token: string) =>
    api.post('/api/auth/verify-email', { token }, { requiresAuth: false }),

  resendVerificationEmail: () =>
    api.post('/api/auth/resend-verification'),

  getCaptcha: () =>
    api.get('/api/auth/captcha', { requiresAuth: false }),
};

// Trading API endpoints
export const tradingApi = {
  getPortfolio: () => api.get('/api/trading/portfolio'),
  getPositions: () => api.get('/api/trading/positions'),
  getOrders: () => api.get('/api/trading/orders'),
  placeOrder: (orderData: any) => api.post('/api/trading/orders', orderData),
  cancelOrder: (orderId: string) => api.delete(`/api/trading/orders/${orderId}`),
  getTransactions: () => api.get('/api/trading/transactions'),
};

// Market data API endpoints
export const marketApi = {
  getAssets: () => api.get('/api/market/assets'),
  getAssetPrice: (symbol: string) => api.get(`/api/market/assets/${symbol}/price`),
  getMarketData: (symbol: string, timeframe?: string) => 
    api.get(`/api/market/data/${symbol}${timeframe ? `?timeframe=${timeframe}` : ''}`),
  getNews: () => api.get('/api/market/news'),
  getWatchlist: () => api.get('/api/market/watchlist'),
  addToWatchlist: (symbol: string) => api.post('/api/market/watchlist', { symbol }),
  removeFromWatchlist: (symbol: string) => api.delete(`/api/market/watchlist/${symbol}`),
};

// Analytics API endpoints
export const analyticsApi = {
  getPerformanceMetrics: () => api.get('/api/analytics/performance'),
  getRiskMetrics: () => api.get('/api/analytics/risk'),
  getBacktestResults: (strategyId: string) => api.get(`/api/analytics/backtest/${strategyId}`),
  runBacktest: (strategyData: any) => api.post('/api/analytics/backtest', strategyData),
};

// Chart Analysis API endpoints
export const chartAnalysisApi = {
  analyzeChart: (formData: FormData) => {
    return apiClient.request('/api/chart-analysis/analyze', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
      requiresAuth: true,
    });
  },
  
  analyzeChartBase64: (data: {
    image_data: string;
    symbol: string;
    timeframe?: string;
    custom_indicators?: string[];
    enable_forecasting?: boolean;
    enable_execution?: boolean;
  }) => api.post('/api/chart-analysis/analyze-base64', data),
  
  batchAnalyze: (data: {
    charts: Array<{
      image_data: string;
      symbol: string;
      timeframe?: string;
      custom_indicators?: string[];
    }>;
    enable_forecasting?: boolean;
    enable_execution?: boolean;
  }) => api.post('/api/chart-analysis/batch-analyze', data),
  
  getPipelineHealth: () => api.get('/api/chart-analysis/health'),
  
  getSupportedIndicators: () => api.get('/api/chart-analysis/indicators'),
  
  getPipelineMetrics: () => api.get('/api/chart-analysis/metrics'),
  
  updatePipelineConfig: (config: {
    timeout_seconds?: number;
    confidence_threshold?: number;
    enable_caching?: boolean;
    max_batch_size?: number;
  }) => api.put('/api/chart-analysis/config', config),
  
  getPipelineStatus: () => api.get('/api/chart-analysis/status'),
};

// Screen Analysis API endpoints
export const screenAnalysisApi = {
  captureRegion: (data: any) => api.post('/api/screen-analysis/capture-region', data),
  captureFullScreen: (data: any) => api.post('/api/screen-analysis/capture-fullscreen', data),
  analyzeMarket: (data: any) => api.post('/api/screen-analysis/analyze-market', data),
  enhancedAnalysis: (data: any) => api.post('/api/screen-analysis/enhanced-analysis', data),
  getCaptureHistory: () => api.get('/api/screen-analysis/capture-history'),
  clearCaptureHistory: () => api.delete('/api/screen-analysis/capture-history'),
};

export default api;