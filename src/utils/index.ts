// Class name utility for merging Tailwind classes
export const cn = (...classes: (string | undefined | null | false)[]): string => {
  return classes.filter(Boolean).join(' ');
};

// Formatting utilities
export const formatCurrency = (value: number, currency = 'USD', decimals = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatNumber = (value: number, decimals = 2): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatPercentage = (value: number, decimals = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value / 100);
};

export const formatLargeNumber = (value: number): string => {
  const absValue = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  
  if (absValue >= 1e12) {
    return `${sign}${(absValue / 1e12).toFixed(2)}T`;
  } else if (absValue >= 1e9) {
    return `${sign}${(absValue / 1e9).toFixed(2)}B`;
  } else if (absValue >= 1e6) {
    return `${sign}${(absValue / 1e6).toFixed(2)}M`;
  } else if (absValue >= 1e3) {
    return `${sign}${(absValue / 1e3).toFixed(2)}K`;
  }
  return `${sign}${absValue.toFixed(2)}`;
};

export const formatDate = (date: string | Date, format: 'short' | 'long' | 'relative' = 'short'): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  
  if (format === 'relative') {
    return formatRelativeTime(dateObj);
  }
  
  const options: Intl.DateTimeFormatOptions = 
    format === 'long'
      ? { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' }
      : { year: 'numeric', month: 'short', day: 'numeric' };
  
  return new Intl.DateTimeFormat('en-US', options).format(dateObj);
};

export const formatRelativeTime = (date: Date | string): string => {
  const now = new Date();
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const diffInSeconds = Math.floor((now.getTime() - dateObj.getTime()) / 1000);
  
  if (diffInSeconds < 60) {
    return 'just now';
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes}m ago`;
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours}h ago`;
  } else if (diffInSeconds < 604800) {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days}d ago`;
  } else {
    return formatDate(dateObj, 'short');
  }
};

// Color utilities
export const getChangeColor = (change: number): string => {
  if (change > 0) return 'text-green-500';
  if (change < 0) return 'text-red-500';
  return 'text-gray-500';
};

export const getChangeColorHex = (change: number): string => {
  if (change > 0) return '#10b981';
  if (change < 0) return '#ef4444';
  return '#6b7280';
};

export const getSentimentColor = (sentiment: 'positive' | 'neutral' | 'negative'): string => {
  switch (sentiment) {
    case 'positive':
      return 'text-green-500';
    case 'negative':
      return 'text-red-500';
    default:
      return 'text-gray-500';
  }
};

// Mathematical utilities
export const calculatePercentageChange = (current: number, previous: number): number => {
  if (previous === 0) return 0;
  return ((current - previous) / previous) * 100;
};

export const calculateMovingAverage = (data: number[], period: number): number[] => {
  const result: number[] = [];
  
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((acc, val) => acc + val, 0);
    result.push(sum / period);
  }
  
  return result;
};

export const calculateRSI = (prices: number[], period = 14): number[] => {
  const gains: number[] = [];
  const losses: number[] = [];
  const rsi: number[] = [];
  
  // Calculate price changes
  for (let i = 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }
  
  // Calculate RSI
  for (let i = period - 1; i < gains.length; i++) {
    const avgGain = gains.slice(i - period + 1, i + 1).reduce((sum, gain) => sum + gain, 0) / period;
    const avgLoss = losses.slice(i - period + 1, i + 1).reduce((sum, loss) => sum + loss, 0) / period;
    
    if (avgLoss === 0) {
      rsi.push(100);
    } else {
      const rs = avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }
  }
  
  return rsi;
};

export const calculateBollingerBands = (
  prices: number[],
  period = 20,
  stdDev = 2
): { upper: number[]; middle: number[]; lower: number[] } => {
  const middle = calculateMovingAverage(prices, period);
  const upper: number[] = [];
  const lower: number[] = [];
  
  for (let i = period - 1; i < prices.length; i++) {
    const slice = prices.slice(i - period + 1, i + 1);
    const mean = slice.reduce((sum, price) => sum + price, 0) / period;
    const variance = slice.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / period;
    const standardDeviation = Math.sqrt(variance);
    
    const middleValue = middle[i - period + 1];
    upper.push(middleValue + (stdDev * standardDeviation));
    lower.push(middleValue - (stdDev * standardDeviation));
  }
  
  return { upper, middle, lower };
};

export const calculateVolatility = (prices: number[], period = 30): number => {
  if (prices.length < period) return 0;
  
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push(Math.log(prices[i] / prices[i - 1]));
  }
  
  const recentReturns = returns.slice(-period);
  const mean = recentReturns.reduce((sum, ret) => sum + ret, 0) / recentReturns.length;
  const variance = recentReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / recentReturns.length;
  
  return Math.sqrt(variance * 252) * 100; // Annualized volatility as percentage
};

// Data manipulation utilities
export const groupBy = <T>(array: T[], key: keyof T): Record<string, T[]> => {
  return array.reduce((groups, item) => {
    const group = String(item[key]);
    groups[group] = groups[group] || [];
    groups[group].push(item);
    return groups;
  }, {} as Record<string, T[]>);
};

export const sortBy = <T>(array: T[], key: keyof T, order: 'asc' | 'desc' = 'asc'): T[] => {
  return [...array].sort((a, b) => {
    const aValue = a[key];
    const bValue = b[key];
    
    if (aValue < bValue) return order === 'asc' ? -1 : 1;
    if (aValue > bValue) return order === 'asc' ? 1 : -1;
    return 0;
  });
};

export const filterBy = <T>(array: T[], filters: Partial<Record<keyof T, any>>): T[] => {
  return array.filter(item => {
    return Object.entries(filters).every(([key, value]) => {
      if (value === undefined || value === null || value === '') return true;
      if (value === 'all') return true;
      return item[key as keyof T] === value;
    });
  });
};

export const searchInObject = <T>(obj: T, query: string, fields: (keyof T)[]): boolean => {
  const lowerQuery = query.toLowerCase();
  return fields.some(field => {
    const value = obj[field];
    if (typeof value === 'string') {
      return value.toLowerCase().includes(lowerQuery);
    }
    return false;
  });
};

// Validation utilities
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const isValidPassword = (password: string): boolean => {
  // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
  const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;
  return passwordRegex.test(password);
};

export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

// Local storage utilities
export const setLocalStorage = (key: string, value: any): void => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error('Error setting localStorage:', error);
  }
};

export const getLocalStorage = <T>(key: string, defaultValue: T): T => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error('Error getting localStorage:', error);
    return defaultValue;
  }
};

export const removeLocalStorage = (key: string): void => {
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.error('Error removing localStorage:', error);
  }
};

// Debounce utility
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void => {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// Throttle utility
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void => {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

// Error handling utilities
export const handleError = (error: any): string => {
  if (error?.response?.data?.message) {
    return error.response.data.message;
  }
  if (error?.message) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'An unexpected error occurred';
};

export const logError = (error: any, context?: string): void => {
  console.error(`Error${context ? ` in ${context}` : ''}:`, error);
  
  // In production, you might want to send errors to a logging service
  if (process.env.NODE_ENV === 'production') {
    // Send to error tracking service (e.g., Sentry, LogRocket)
  }
};

// Chart utilities
export const generateChartColors = (count: number): string[] => {
  const colors = [
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // yellow
    '#ef4444', // red
    '#8b5cf6', // purple
    '#06b6d4', // cyan
    '#f97316', // orange
    '#84cc16', // lime
    '#ec4899', // pink
    '#6b7280', // gray
  ];
  
  const result = [];
  for (let i = 0; i < count; i++) {
    result.push(colors[i % colors.length]);
  }
  
  return result;
};

export const interpolateColor = (color1: string, color2: string, factor: number): string => {
  const hex1 = color1.replace('#', '');
  const hex2 = color2.replace('#', '');
  
  const r1 = parseInt(hex1.substr(0, 2), 16);
  const g1 = parseInt(hex1.substr(2, 2), 16);
  const b1 = parseInt(hex1.substr(4, 2), 16);
  
  const r2 = parseInt(hex2.substr(0, 2), 16);
  const g2 = parseInt(hex2.substr(2, 2), 16);
  const b2 = parseInt(hex2.substr(4, 2), 16);
  
  const r = Math.round(r1 + (r2 - r1) * factor);
  const g = Math.round(g1 + (g2 - g1) * factor);
  const b = Math.round(b1 + (b2 - b1) * factor);
  
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
};

// Asset utilities
export const getAssetIcon = (symbol: string): string => {
  const iconMap: Record<string, string> = {
    BTC: '₿',
    ETH: 'Ξ',
    USD: '$',
    EUR: '€',
    GBP: '£',
    JPY: '¥',
    GOLD: '🥇',
    SILVER: '🥈',
  };
  
  return iconMap[symbol.toUpperCase()] || symbol.charAt(0).toUpperCase();
};

export const getAssetCategory = (symbol: string): 'crypto' | 'stocks' | 'forex' | 'commodities' => {
  const cryptoSymbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'COMP'];
  const forexSymbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'];
  const commoditySymbols = ['GOLD', 'SILVER', 'OIL', 'COPPER', 'WHEAT', 'CORN'];
  
  const upperSymbol = symbol.toUpperCase();
  
  if (cryptoSymbols.includes(upperSymbol)) return 'crypto';
  if (forexSymbols.includes(upperSymbol)) return 'forex';
  if (commoditySymbols.includes(upperSymbol)) return 'commodities';
  
  return 'stocks';
};

// Time utilities
export const getTimeframeLabel = (timeframe: string): string => {
  const labels: Record<string, string> = {
    '1m': '1 Minute',
    '5m': '5 Minutes',
    '15m': '15 Minutes',
    '1h': '1 Hour',
    '4h': '4 Hours',
    '1d': '1 Day',
    '1w': '1 Week',
    '1M': '1 Month',
  };
  
  return labels[timeframe] || timeframe;
};

export const getTimeframeMilliseconds = (timeframe: string): number => {
  const milliseconds: Record<string, number> = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
    '1w': 7 * 24 * 60 * 60 * 1000,
    '1M': 30 * 24 * 60 * 60 * 1000,
  };
  
  return milliseconds[timeframe] || 60 * 1000;
};

// Export all utilities
export default {
  formatCurrency,
  formatNumber,
  formatPercentage,
  formatLargeNumber,
  formatDate,
  formatRelativeTime,
  getChangeColor,
  getChangeColorHex,
  getSentimentColor,
  calculatePercentageChange,
  calculateMovingAverage,
  calculateRSI,
  calculateBollingerBands,
  calculateVolatility,
  groupBy,
  sortBy,
  filterBy,
  searchInObject,
  isValidEmail,
  isValidPassword,
  isValidUrl,
  setLocalStorage,
  getLocalStorage,
  removeLocalStorage,
  debounce,
  throttle,
  handleError,
  logError,
  generateChartColors,
  interpolateColor,
  getAssetIcon,
  getAssetCategory,
  getTimeframeLabel,
  getTimeframeMilliseconds,
};