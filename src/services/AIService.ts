import axios from 'axios';
import { io, Socket } from 'socket.io-client';

// Types for AI services
export interface MarketPrediction {
  symbol: string;
  timeframe: '1h' | '4h' | '1d' | '1w';
  direction: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  targetPrice: number;
  stopLoss: number;
  reasoning: string;
  technicalIndicators: {
    rsi: number;
    macd: { signal: number; histogram: number };
    bollinger: { upper: number; lower: number; middle: number };
    support: number[];
    resistance: number[];
  };
  fundamentalFactors: string[];
  riskScore: number;
  expectedReturn: number;
  timeHorizon: string;
}

export interface TradingStrategy {
  id: string;
  name: string;
  description: string;
  type: 'momentum' | 'mean_reversion' | 'breakout' | 'arbitrage' | 'ml_ensemble';
  riskLevel: 'low' | 'medium' | 'high';
  status?: 'active' | 'paused' | 'stopped';
  expectedReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  winRate: number;
  avgHoldingPeriod: string;
  assets: string[];
  parameters: Record<string, any>;
  performance: {
    totalReturn: number;
    annualizedReturn: number;
    volatility: number;
    maxDrawdown: number;
    calmarRatio: number;
    sortinoRatio: number;
    trades: number;
    winningTrades: number;
    avgWin: number;
    avgLoss: number;
  };
  signals: TradingSignal[];
}

export interface TradingSignal {
  id: string;
  strategyId: string;
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  strength: number;
  confidence: number;
  price: number;
  timestamp: Date;
  reasoning: string;
  technicalAnalysis: {
    indicators: Record<string, number>;
    patterns: string[];
    support: number;
    resistance: number;
  };
  fundamentalAnalysis?: {
    score: number;
    factors: string[];
  };
  riskMetrics: {
    volatility: number;
    beta: number;
    var: number;
    expectedShortfall: number;
  };
}

export interface AIInsight {
  id: string;
  type: 'market_outlook' | 'portfolio_optimization' | 'risk_alert' | 'opportunity';
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  confidence: number;
  actionable: boolean;
  recommendations: string[];
  affectedAssets: string[];
  timeframe: string;
  createdAt: Date;
  metadata: Record<string, any>;
}

export interface PortfolioOptimization {
  currentAllocation: Record<string, number>;
  optimizedAllocation: Record<string, number>;
  expectedReturn: number;
  expectedRisk: number;
  sharpeRatio: number;
  diversificationRatio: number;
  recommendations: {
    action: 'buy' | 'sell' | 'rebalance';
    asset: string;
    amount: number;
    reasoning: string;
  }[];
  riskMetrics: {
    var: number;
    cvar: number;
    maxDrawdown: number;
    correlationMatrix: Record<string, Record<string, number>>;
  };
}

export interface SentimentAnalysis {
  symbol: string;
  overall: 'bullish' | 'bearish' | 'neutral';
  score: number;
  sources: {
    news: { score: number; articles: number };
    social: { score: number; mentions: number };
    analyst: { score: number; ratings: number };
    options: { score: number; putCallRatio: number };
  };
  trends: {
    daily: number[];
    weekly: number[];
    monthly: number[];
  };
  keywords: { word: string; sentiment: number; frequency: number }[];
  events: {
    date: Date;
    event: string;
    impact: number;
  }[];
}

class AIService {
  private baseURL: string;
  private socket: Socket | null = null;
  private apiKey: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_AI_API_URL || 'http://localhost:8001';
    this.apiKey = process.env.REACT_APP_AI_API_KEY || '';
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    this.socket = io(`${this.baseURL}/ai`, {
      auth: {
        token: this.apiKey,
      },
      transports: ['websocket'],
    });

    this.socket.on('connect', () => {
      console.log('Connected to AI service');
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from AI service');
    });

    this.socket.on('prediction_update', (data: MarketPrediction) => {
      // Handle real-time prediction updates
      this.handlePredictionUpdate(data);
    });

    this.socket.on('signal_generated', (data: TradingSignal) => {
      // Handle new trading signals
      this.handleNewSignal(data);
    });

    this.socket.on('insight_generated', (data: AIInsight) => {
      // Handle new AI insights
      this.handleNewInsight(data);
    });
  }

  // Market Predictions
  async getMarketPrediction(symbol: string, timeframe: string = '1d'): Promise<MarketPrediction> {
    try {
      const response = await axios.get(`${this.baseURL}/api/predictions/${symbol}`, {
        params: { timeframe },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get market prediction:', error);
      throw error;
    }
  }

  async getBatchPredictions(symbols: string[], timeframe: string = '1d'): Promise<MarketPrediction[]> {
    try {
      const response = await axios.post(`${this.baseURL}/api/predictions/batch`, {
        symbols,
        timeframe,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get batch predictions:', error);
      throw error;
    }
  }

  // Trading Strategies
  async getAvailableStrategies(): Promise<TradingStrategy[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/strategies`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get strategies:', error);
      throw error;
    }
  }

  async getStrategyPerformance(strategyId: string, period: string = '1y'): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/api/strategies/${strategyId}/performance`, {
        params: { period },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get strategy performance:', error);
      throw error;
    }
  }

  async createCustomStrategy(config: any): Promise<TradingStrategy> {
    try {
      const response = await axios.post(`${this.baseURL}/api/strategies/custom`, config, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to create custom strategy:', error);
      throw error;
    }
  }

  async backtestStrategy(strategyId: string, config: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/strategies/${strategyId}/backtest`, config, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to backtest strategy:', error);
      throw error;
    }
  }

  // Trading Signals
  async getActiveSignals(filters?: any): Promise<TradingSignal[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/signals`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get active signals:', error);
      throw error;
    }
  }

  async getSignalHistory(symbol: string, limit: number = 50): Promise<TradingSignal[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/signals/history/${symbol}`, {
        params: { limit },
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get signal history:', error);
      throw error;
    }
  }

  // Portfolio Optimization
  async optimizePortfolio(portfolio: any, constraints?: any): Promise<PortfolioOptimization> {
    try {
      const response = await axios.post(`${this.baseURL}/api/portfolio/optimize`, {
        portfolio,
        constraints,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to optimize portfolio:', error);
      throw error;
    }
  }

  async getRiskAnalysis(portfolio: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/portfolio/risk`, portfolio, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get risk analysis:', error);
      throw error;
    }
  }

  // Sentiment Analysis
  async getSentimentAnalysis(symbol: string): Promise<SentimentAnalysis> {
    try {
      const response = await axios.get(`${this.baseURL}/api/sentiment/${symbol}`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get sentiment analysis:', error);
      throw error;
    }
  }

  async getMarketSentiment(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/api/sentiment/market`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get market sentiment:', error);
      throw error;
    }
  }

  // AI Insights
  async getAIInsights(filters?: any): Promise<AIInsight[]> {
    try {
      const response = await axios.get(`${this.baseURL}/api/insights`, {
        params: filters,
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get AI insights:', error);
      throw error;
    }
  }

  async generateInsight(type: string, context: any): Promise<AIInsight> {
    try {
      const response = await axios.post(`${this.baseURL}/api/insights/generate`, {
        type,
        context,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to generate insight:', error);
      throw error;
    }
  }

  // Natural Language Processing
  async analyzeText(text: string, type: 'sentiment' | 'entities' | 'summary' = 'sentiment'): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/nlp/analyze`, {
        text,
        type,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to analyze text:', error);
      throw error;
    }
  }

  async generateExplanation(data: any, type: 'prediction' | 'signal' | 'insight'): Promise<string> {
    try {
      const response = await axios.post(`${this.baseURL}/api/nlp/explain`, {
        data,
        type,
      }, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data.explanation;
    } catch (error) {
      console.error('Failed to generate explanation:', error);
      throw error;
    }
  }

  // Model Management
  async getModelStatus(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/api/models/status`, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get model status:', error);
      throw error;
    }
  }

  async retrainModel(modelId: string, config?: any): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/models/${modelId}/retrain`, config, {
        headers: { Authorization: `Bearer ${this.apiKey}` },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to retrain model:', error);
      throw error;
    }
  }

  // Real-time subscriptions
  subscribeToPredictions(symbols: string[], callback: (prediction: MarketPrediction) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_predictions', { symbols });
      this.socket.on('prediction_update', callback);
    }
  }

  subscribeToSignals(strategies: string[], callback: (signal: TradingSignal) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_signals', { strategies });
      this.socket.on('signal_generated', callback);
    }
  }

  subscribeToInsights(callback: (insight: AIInsight) => void) {
    if (this.socket) {
      this.socket.emit('subscribe_insights');
      this.socket.on('insight_generated', callback);
    }
  }

  // Event handlers
  private handlePredictionUpdate(prediction: MarketPrediction) {
    // Dispatch to Redux store or handle as needed
    console.log('New prediction:', prediction);
  }

  private handleNewSignal(signal: TradingSignal) {
    // Dispatch to Redux store or handle as needed
    console.log('New signal:', signal);
  }

  private handleNewInsight(insight: AIInsight) {
    // Dispatch to Redux store or handle as needed
    console.log('New insight:', insight);
  }

  // Cleanup
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const aiService = new AIService();
export default aiService;