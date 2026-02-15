import axios from 'axios';

const client = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8080/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PriceData {
  symbol: string;
  price: string;
  timestamp: string;
}

export interface NewsArticle {
  id: string;
  title: string;
  content: string;
  source: string;
  url: string;
  timestamp: string;
  sentiment: number;
  tags: string[];
}

export interface ScreenerResult {
  id: string;
  ticker: string;
  company: string;
  sector: string;
  industry: string;
  country: string;
  market_cap: string;
  pe: string;
  price: string;
  change: string;
  volume: string;
  dividend_yield: string;
  eps: string;
  revenue: string;
  debt: string;
  roe: string;
  profit_margin: string;
  book_value: string;
  strategy: string;
  fetched_at: string;
}

export interface InsiderTrade {
  id: string;
  ticker: string;
  owner: string;
  relationship: string;
  date: string;
  transaction: string;
  cost: number;
  shares: number;
  value: number;
  total_shares: number;
  sec_form_4: string;
  fetched_at: string;
}

export interface SectorPerformance {
  id: string;
  name: string;
  change: number;
  volume: number;
  stocks: string;
  market_cap: number;
  pe: number;
  fetched_at: string;
}

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface FundamentalsRecord {
  id: string;
  ticker: string;
  period: string;
  timeframe: string;
  revenue: number;
  net_income: number;
  total_assets: number;
  total_liabilities: number;
  operating_cashflow: number;
  metrics?: Record<string, any>; // Comprehensive metrics map
  fetched_at: string;
}

export const getStockCandles = async (symbol: string, timeframe: string): Promise<Candle[]> => {
  const response = await client.get(`/market/candles/${symbol}?timeframe=${timeframe}`);
  return response.data || [];
};

export const getFundamentals = async (symbol: string, timeframe: string) => {
  const response = await client.get(`/fundamentals/${symbol}?timeframe=${timeframe}`);
  return response.data as FundamentalsRecord;
};

export const getPrice = async (symbol: string): Promise<PriceData> => {
  const response = await client.get(`/market/price/${symbol}`);
  return response.data;
};

export const getMarketMovers = async () => {
  const response = await client.get('/market/movers');
  return response.data;
};

export const getLatestNews = async (limit: number = 10): Promise<NewsArticle[]> => {
  const response = await client.get(`/news/latest?limit=${limit}`);
  return response.data || [];
};

export const getScreenerResults = async (strategy: string = '', limit: number = 50): Promise<ScreenerResult[]> => {
  const response = await client.get(`/screener/?strategy=${strategy}&limit=${limit}`);
  return response.data || [];
};

export const getInsiderTrades = async (limit: number = 50): Promise<InsiderTrade[]> => {
  const response = await client.get(`/insider/?limit=${limit}`);
  return response.data || [];
};

export const getSectorPerformance = async (): Promise<SectorPerformance[]> => {
  const response = await client.get(`/sector/`);
  return response.data || [];
};

// Auth API
export const login = async (credentials: any) => {
  const response = await client.post('/auth/login', credentials);
  return response.data;
};

export const register = async (userData: any) => {
  const response = await client.post('/auth/register', userData);
  return response.data;
};

export default client;
