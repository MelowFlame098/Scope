'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertTriangle, 
  Target, 
  Shield, 
  Brain,
  X,
  ChevronUp,
  ChevronDown,
  Eye,
  EyeOff
} from 'lucide-react';
import { aiAnalysisService } from '../services/aiAnalysisService';
import { FinR1Output } from '../types/ai';
import LoadingSpinner from './ui/LoadingSpinner';

interface AnalysisResult {
  success: boolean;
  symbol: string;
  timeframe: string;
  strategy_recommendations?: {
    primary_recommendation: string;
    confidence: number;
    strategies_count: number;
    market_sentiment?: any[];
    scraped_signals?: any[];
  };
  execution_signals?: {
    action: string;
    confidence: number;
    signal_strength: string;
    position_size: number;
    stop_loss?: number;
    take_profit?: number;
    reasoning: string;
  }[];
  forecasts?: {
    price_forecast?: {
      direction: string;
      confidence: number;
      target_price?: number;
    };
    volatility_forecast?: {
      level: string;
      confidence: number;
    };
  };
  market_analysis?: {
    overall_recommendation?: {
      recommendation: string;
      confidence: number;
      sentiment_score: number;
    };
    sentiment_analysis?: {
      average_sentiment: number;
      confidence: number;
      data_points: number;
    };
    trading_signals?: any[];
  };
}

interface ChartOverlayProps {
  symbol: string;
  isVisible: boolean;
  onToggleVisibility: () => void;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

const ChartOverlay: React.FC<ChartOverlayProps> = ({
  symbol,
  isVisible,
  onToggleVisibility,
  position = 'top-right'
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'signals' | 'sentiment' | 'forecast'>('signals');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        setLoading(true);
        setError(null);
        const finR1Result = await aiAnalysisService.getAnalysis(symbol);
        
        // Convert FinR1Output to AnalysisResult format
        const analysisResult: AnalysisResult = {
          success: true,
          symbol: symbol,
          timeframe: finR1Result.time_horizon,
          strategy_recommendations: {
            primary_recommendation: finR1Result.primary_recommendation,
            confidence: finR1Result.confidence,
            strategies_count: 1,
            market_sentiment: finR1Result.market_sentiment,
            scraped_signals: finR1Result.scraped_signals
          },
          execution_signals: [{
            action: finR1Result.execution_signal.action,
            confidence: finR1Result.confidence,
            signal_strength: finR1Result.execution_signal.urgency,
            position_size: finR1Result.execution_signal.position_size,
            stop_loss: finR1Result.execution_signal.stop_loss,
            take_profit: finR1Result.execution_signal.take_profit,
            reasoning: finR1Result.execution_signal.reasoning
          }],
          forecasts: {
            price_forecast: {
              direction: finR1Result.primary_recommendation === 'BUY' ? 'UP' : finR1Result.primary_recommendation === 'SELL' ? 'DOWN' : 'NEUTRAL',
              confidence: finR1Result.confidence,
              target_price: finR1Result.target_price
            },
            volatility_forecast: {
              level: finR1Result.risk_level,
              confidence: finR1Result.confidence
            }
          },
          market_analysis: {
            overall_recommendation: {
              recommendation: finR1Result.primary_recommendation,
              confidence: finR1Result.confidence,
              sentiment_score: finR1Result.market_sentiment.length > 0 
                ? finR1Result.market_sentiment.reduce((acc, s) => acc + s.sentiment_score, 0) / finR1Result.market_sentiment.length
                : 0
            },
            sentiment_analysis: {
              average_sentiment: finR1Result.market_sentiment.length > 0 
                ? finR1Result.market_sentiment.reduce((acc, s) => acc + s.sentiment_score, 0) / finR1Result.market_sentiment.length
                : 0,
              confidence: finR1Result.market_sentiment.length > 0 
                ? finR1Result.market_sentiment.reduce((acc, s) => acc + s.confidence, 0) / finR1Result.market_sentiment.length
                : 0,
              data_points: finR1Result.market_sentiment.length
            },
            trading_signals: finR1Result.scraped_signals
          }
        };
        
        setAnalysisResult(analysisResult);
      } catch (err) {
        setError('Failed to load AI analysis');
        console.error('Error fetching analysis:', err);
      } finally {
        setLoading(false);
      }
    };

    if (symbol) {
      fetchAnalysis();
    }
  }, [symbol]);

  const getPositionClasses = () => {
    switch (position) {
      case 'top-left':
        return 'top-4 left-4';
      case 'top-right':
        return 'top-4 right-4';
      case 'bottom-left':
        return 'bottom-4 left-4';
      case 'bottom-right':
        return 'bottom-4 right-4';
      default:
        return 'top-4 right-4';
    }
  };

  const getRecommendationIcon = (recommendation: string) => {
    switch (recommendation?.toLowerCase()) {
      case 'buy':
      case 'strong buy':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'sell':
      case 'strong sell':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case 'hold':
        return <Minus className="h-4 w-4 text-yellow-600" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-gray-600" />;
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation?.toLowerCase()) {
      case 'buy':
      case 'strong buy':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'sell':
      case 'strong sell':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'hold':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getSentimentColor = (sentiment: number) => {
    if (sentiment > 0.3) return 'text-green-600';
    if (sentiment < -0.3) return 'text-red-600';
    return 'text-yellow-600';
  };

  if (!isVisible) {
    return (
      <div className={`absolute ${getPositionClasses()} z-40`}>
        <Button
          onClick={onToggleVisibility}
          size="sm"
          variant="outline"
          className="bg-white/90 backdrop-blur-sm border-gray-200 shadow-lg hover:bg-white"
        >
          <Eye className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className={`absolute ${getPositionClasses()} z-40 w-80`}>
      <Card className="bg-white/95 backdrop-blur-sm border-gray-200 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between p-3 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-blue-600" />
            <span className="font-medium text-sm">AI Analysis</span>
            {symbol && (
              <Badge variant="outline" className="text-xs">
                {symbol}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button
              onClick={() => setIsExpanded(!isExpanded)}
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
            >
              {isExpanded ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </Button>
            <Button
              onClick={onToggleVisibility}
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
            >
              <EyeOff className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <CardContent className="p-3">
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <LoadingSpinner size="md" />
              <span className="ml-2 text-sm text-gray-600">Loading AI analysis...</span>
            </div>
          ) : error || !analysisResult ? (
            <div className="text-center text-red-600">
              <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
              <p className="text-sm">{error || 'Failed to load analysis'}</p>
            </div>
          ) : analysisResult?.success ? (
            <>
              {/* Primary Recommendation */}
              {analysisResult.strategy_recommendations && (
                <div className="mb-3">
                  <div className="flex items-center gap-2 mb-2">
                    {getRecommendationIcon(analysisResult.strategy_recommendations.primary_recommendation)}
                    <Badge className={getRecommendationColor(analysisResult.strategy_recommendations.primary_recommendation)}>
                      {analysisResult.strategy_recommendations.primary_recommendation}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {(analysisResult.strategy_recommendations.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Execution Signals */}
              {analysisResult.execution_signals && analysisResult.execution_signals.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs font-medium text-gray-700 mb-1">Execution Signal</div>
                  <div className="bg-gray-50 p-2 rounded text-xs">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium">
                        {analysisResult.execution_signals[0].action.toUpperCase()}
                      </span>
                      <span className="text-gray-500">
                        {(analysisResult.execution_signals[0].confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    {analysisResult.execution_signals[0].stop_loss && (
                      <div className="flex items-center gap-1 text-red-600">
                        <Shield className="h-3 w-3" />
                        <span>SL: ${analysisResult.execution_signals[0].stop_loss.toFixed(2)}</span>
                      </div>
                    )}
                    {analysisResult.execution_signals[0].take_profit && (
                      <div className="flex items-center gap-1 text-green-600">
                        <Target className="h-3 w-3" />
                        <span>TP: ${analysisResult.execution_signals[0].take_profit.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {isExpanded && (
                <>
                  {/* Tab Navigation */}
                  <div className="flex border-b border-gray-200 mb-3">
                    <button
                      onClick={() => setActiveTab('signals')}
                      className={`px-2 py-1 text-xs font-medium border-b-2 transition-colors ${
                        activeTab === 'signals'
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      Signals
                    </button>
                    <button
                      onClick={() => setActiveTab('sentiment')}
                      className={`px-2 py-1 text-xs font-medium border-b-2 transition-colors ${
                        activeTab === 'sentiment'
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      Sentiment
                    </button>
                    <button
                      onClick={() => setActiveTab('forecast')}
                      className={`px-2 py-1 text-xs font-medium border-b-2 transition-colors ${
                        activeTab === 'forecast'
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700'
                      }`}
                    >
                      Forecast
                    </button>
                  </div>

                  {/* Tab Content */}
                  {activeTab === 'signals' && (
                    <div className="space-y-2">
                      {analysisResult.market_analysis?.trading_signals?.slice(0, 3).map((signal: any, idx: number) => (
                        <div key={idx} className="bg-white border border-gray-200 p-2 rounded text-xs">
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-medium">{signal.type?.toUpperCase()}</span>
                            <span className="text-gray-500">{(signal.strength * 100).toFixed(0)}%</span>
                          </div>
                          <div className="text-gray-600">{signal.source}</div>
                          {signal.reasoning && (
                            <div className="text-gray-500 mt-1 text-xs">{signal.reasoning.slice(0, 60)}...</div>
                          )}
                        </div>
                      )) || (
                        <div className="text-xs text-gray-500 text-center py-2">
                          No trading signals available
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === 'sentiment' && (
                    <div className="space-y-2">
                      {analysisResult.market_analysis?.sentiment_analysis && (
                        <div className="bg-white border border-gray-200 p-2 rounded">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs font-medium">Market Sentiment</span>
                            <span className={`text-xs font-medium ${
                              getSentimentColor(analysisResult.market_analysis.sentiment_analysis.average_sentiment)
                            }`}>
                              {(analysisResult.market_analysis.sentiment_analysis.average_sentiment * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="text-xs text-gray-500">
                            Based on {analysisResult.market_analysis.sentiment_analysis.data_points} data points
                          </div>
                        </div>
                      )}
                      
                      {analysisResult.strategy_recommendations?.market_sentiment?.slice(0, 2).map((sentiment: any, idx: number) => (
                        <div key={idx} className="bg-white border border-gray-200 p-2 rounded text-xs">
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-medium">{sentiment.source}</span>
                            <span className={`font-medium ${getSentimentColor(sentiment.sentiment_score)}`}>
                              {(sentiment.sentiment_score * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-gray-600">{sentiment.summary?.slice(0, 50)}...</div>
                        </div>
                      )) || (
                        <div className="text-xs text-gray-500 text-center py-2">
                          No sentiment data available
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === 'forecast' && (
                    <div className="space-y-2">
                      {analysisResult.forecasts?.price_forecast && (
                        <div className="bg-white border border-gray-200 p-2 rounded">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs font-medium">Price Forecast</span>
                            <span className={`text-xs font-medium ${
                              analysisResult.forecasts.price_forecast.direction === 'up' ? 'text-green-600' : 
                              analysisResult.forecasts.price_forecast.direction === 'down' ? 'text-red-600' : 'text-yellow-600'
                            }`}>
                              {analysisResult.forecasts.price_forecast.direction.toUpperCase()}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500">
                            Confidence: {(analysisResult.forecasts.price_forecast.confidence * 100).toFixed(1)}%
                          </div>
                          {analysisResult.forecasts.price_forecast.target_price && (
                            <div className="text-xs text-gray-600">
                              Target: ${analysisResult.forecasts.price_forecast.target_price.toFixed(2)}
                            </div>
                          )}
                        </div>
                      )}
                      
                      {analysisResult.forecasts?.volatility_forecast && (
                        <div className="bg-white border border-gray-200 p-2 rounded">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs font-medium">Volatility</span>
                            <span className="text-xs font-medium text-blue-600">
                              {analysisResult.forecasts.volatility_forecast.level.toUpperCase()}
                            </span>
                          </div>
                          <div className="text-xs text-gray-500">
                            Confidence: {(analysisResult.forecasts.volatility_forecast.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                      
                      {(!analysisResult.forecasts?.price_forecast && !analysisResult.forecasts?.volatility_forecast) && (
                        <div className="text-xs text-gray-500 text-center py-2">
                          No forecast data available
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </>
          ) : (
            <div className="text-center py-4">
              <AlertTriangle className="h-8 w-8 text-gray-400 mx-auto mb-2" />
              <div className="text-xs text-gray-500">
                {analysisResult?.success === false ? 'Analysis failed' : 'No analysis data'}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default ChartOverlay;