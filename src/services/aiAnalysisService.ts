import { FinR1Output, MarketSentiment, TradingSignal } from '../types/ai';

// Mock AI analysis data for demonstration
const generateMockAnalysis = (symbol: string): FinR1Output => {
  const mockSentiments: MarketSentiment[] = [
    {
      symbol,
      sentiment_score: 0.75,
      source: 'Reddit',
      summary: 'Bullish sentiment on recent earnings beat',
      timestamp: new Date().toISOString(),
      confidence: 0.85
    },
    {
      symbol,
      sentiment_score: 0.65,
      source: 'Twitter',
      summary: 'Positive technical analysis discussions',
      timestamp: new Date().toISOString(),
      confidence: 0.72
    }
  ];

  const mockSignals: TradingSignal[] = [
    {
      signal_type: 'BUY',
      strength: 0.8,
      source: 'Technical Analysis',
      reasoning: 'Strong bullish momentum with RSI oversold recovery',
      target_price: 155.50,
      stop_loss: 142.30,
      timestamp: new Date().toISOString(),
      confidence: 0.78
    },
    {
      signal_type: 'HOLD',
      strength: 0.6,
      source: 'Fundamental Analysis',
      reasoning: 'Fair valuation with moderate growth prospects',
      target_price: 148.00,
      stop_loss: 135.00,
      timestamp: new Date().toISOString(),
      confidence: 0.65
    }
  ];

  return {
    primary_recommendation: 'BUY',
    confidence: 0.82,
    reasoning: 'Strong technical indicators combined with positive market sentiment suggest upward momentum. Recent earnings beat and improving fundamentals support bullish outlook.',
    target_price: 155.50,
    stop_loss: 142.30,
    risk_level: 'MEDIUM',
    time_horizon: '1-3 months',
    key_factors: [
      'Technical breakout above resistance',
      'Positive earnings surprise',
      'Strong sector momentum',
      'Improving market sentiment'
    ],
    execution_signal: {
      action: 'BUY',
      urgency: 'MEDIUM',
      position_size: 0.05,
      entry_price: 145.20,
      stop_loss: 142.30,
      take_profit: 155.50,
      reasoning: 'Enter on current momentum with defined risk management'
    },
    market_sentiment: mockSentiments,
    scraped_signals: mockSignals,
    metadata: {
      analysis_timestamp: new Date().toISOString(),
      model_version: 'fin-r1-v2.1',
      data_sources: ['technical', 'fundamental', 'sentiment', 'news'],
      processing_time_ms: 1250,
      market_sentiment_count: mockSentiments.length,
      scraped_signals_count: mockSignals.length
    }
  };
};

export const aiAnalysisService = {
  async getAnalysis(symbol: string): Promise<FinR1Output> {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // In a real implementation, this would call the backend API
    // return fetch(`/api/chart-analysis/${symbol}`).then(res => res.json());
    
    return generateMockAnalysis(symbol);
  },

  async getMarketSentiment(symbol: string): Promise<MarketSentiment[]> {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return [
      {
        symbol,
        sentiment_score: Math.random() * 2 - 1, // -1 to 1
        source: 'Reddit',
        summary: 'Mixed sentiment with bullish bias on recent developments',
        timestamp: new Date().toISOString(),
        confidence: 0.75 + Math.random() * 0.2
      },
      {
        symbol,
        sentiment_score: Math.random() * 2 - 1,
        source: 'Twitter',
        summary: 'Technical analysis discussions trending positive',
        timestamp: new Date().toISOString(),
        confidence: 0.65 + Math.random() * 0.25
      }
    ];
  },

  async getTradingSignals(symbol: string): Promise<TradingSignal[]> {
    await new Promise(resolve => setTimeout(resolve, 750));
    
    const signals: ('BUY' | 'SELL' | 'HOLD')[] = ['BUY', 'SELL', 'HOLD'];
    
    return [
      {
        signal_type: signals[Math.floor(Math.random() * signals.length)],
        strength: 0.5 + Math.random() * 0.5,
        source: 'Technical Analysis',
        reasoning: 'RSI and MACD indicators suggest momentum shift',
        target_price: 150 + Math.random() * 20,
        stop_loss: 140 + Math.random() * 10,
        timestamp: new Date().toISOString(),
        confidence: 0.6 + Math.random() * 0.3
      }
    ];
  }
};