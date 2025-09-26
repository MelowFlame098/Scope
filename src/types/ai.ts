// AI Analysis Types

export interface MarketSentiment {
  symbol: string;
  sentiment_score: number; // -1 to 1 scale
  source: string;
  summary: string;
  timestamp: string;
  confidence: number;
}

export interface TradingSignal {
  signal_type: 'BUY' | 'SELL' | 'HOLD';
  strength: number; // 0 to 1 scale
  source: string;
  reasoning: string;
  target_price?: number;
  stop_loss?: number;
  timestamp: string;
  confidence: number;
}

export interface ExecutionSignal {
  action: 'BUY' | 'SELL' | 'HOLD';
  urgency: 'LOW' | 'MEDIUM' | 'HIGH';
  position_size: number; // 0 to 1 scale
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  reasoning: string;
}

export interface FinR1Output {
  primary_recommendation: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  target_price: number;
  stop_loss: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  time_horizon: string;
  key_factors: string[];
  execution_signal: ExecutionSignal;
  market_sentiment: MarketSentiment[];
  scraped_signals: TradingSignal[];
  metadata: {
    analysis_timestamp: string;
    model_version: string;
    data_sources: string[];
    processing_time_ms: number;
    market_sentiment_count: number;
    scraped_signals_count: number;
  };
}

export interface ChartAnalysisResult {
  symbol: string;
  analysis: FinR1Output;
  timestamp: string;
  status: 'loading' | 'success' | 'error';
  error?: string;
}