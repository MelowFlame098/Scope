'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  CogIcon,
  ChartBarIcon,
  LightBulbIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { aiService, TradingStrategy, MarketPrediction, AIInsight, TradingSignal } from '../services/AIService';

interface AITradingStrategiesProps {
  className?: string;
}

const AITradingStrategies: React.FC<AITradingStrategiesProps> = ({ className }) => {
  const [activeTab, setActiveTab] = useState<'strategies' | 'predictions' | 'signals' | 'insights'>('strategies');
  const [strategies, setStrategies] = useState<TradingStrategy[]>([]);
  const [predictions, setPredictions] = useState<MarketPrediction[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [insights, setInsights] = useState<AIInsight[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [strategyPerformance, setStrategyPerformance] = useState<any[]>([]);
  const [modelMetrics, setModelMetrics] = useState<any>(null);

  useEffect(() => {
    loadInitialData();
    setupRealTimeUpdates();

    return () => {
      aiService.disconnect();
    };
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadStrategies(),
        loadPredictions(),
        loadSignals(),
        loadInsights(),
        loadModelMetrics(),
      ]);
    } catch (error) {
      console.error('Failed to load AI data:', error);
    } finally {
      setLoading(false);
    }
  };

  const setupRealTimeUpdates = () => {
    aiService.subscribeToSignals(['momentum', 'mean_reversion'], (signal: TradingSignal) => {
      setSignals(prev => [signal, ...prev.slice(0, 49)]);
    });

    aiService.subscribeToInsights((insight: AIInsight) => {
      setInsights(prev => [insight, ...prev.slice(0, 19)]);
    });
  };

  const loadStrategies = async () => {
    try {
      const strategiesData = await aiService.getAvailableStrategies();
      setStrategies(strategiesData);
      
      // Load performance data for the first strategy
      if (strategiesData.length > 0) {
        setSelectedStrategy(strategiesData[0].id);
        const performance = await aiService.getStrategyPerformance(strategiesData[0].id, '1M');
        setStrategyPerformance(performance);
      }
    } catch (error) {
      console.error('Failed to load strategies:', error);
      // Mock data for demonstration
      setStrategies([
        {
          id: 'momentum-ml',
          name: 'ML Momentum Strategy',
          description: 'Machine learning-based momentum strategy using LSTM networks',
          type: 'momentum',
          riskLevel: 'medium',
          expectedReturn: 0.12,
          maxDrawdown: 0.08,
          sharpeRatio: 1.8,
          winRate: 0.68,
          avgHoldingPeriod: '3.2 days',
          assets: ['AAPL', 'GOOGL', 'MSFT'],
          parameters: {},
          performance: {
            totalReturn: 0.125,
            annualizedReturn: 0.12,
            volatility: 0.18,
            maxDrawdown: 0.08,
            calmarRatio: 1.5,
            sortinoRatio: 1.8,
            trades: 156,
            winningTrades: 106,
            avgWin: 0.025,
            avgLoss: -0.015,
          },
          signals: [],
        },
        {
          id: 'mean-reversion-ai',
          name: 'AI Mean Reversion',
          description: 'Deep learning mean reversion strategy with sentiment analysis',
          type: 'mean_reversion',
          riskLevel: 'low',
          expectedReturn: 0.095,
          maxDrawdown: 0.06,
          sharpeRatio: 1.32,
          winRate: 0.72,
          avgHoldingPeriod: '2.8 days',
          assets: ['SPY', 'QQQ', 'IWM'],
          parameters: {},
          performance: {
            totalReturn: 0.095,
            annualizedReturn: 0.09,
            volatility: 0.15,
            maxDrawdown: 0.06,
            calmarRatio: 1.5,
            sortinoRatio: 1.6,
            trades: 203,
            winningTrades: 146,
            avgWin: 0.018,
            avgLoss: -0.012,
          },
          signals: [],
        },
      ]);
    }
  };

  const loadPredictions = async () => {
    try {
      const predictionsData = await aiService.getBatchPredictions(['AAPL', 'MSFT', 'GOOGL'], '1d');
      setPredictions(predictionsData);
    } catch (error) {
      console.error('Failed to load predictions:', error);
      // Mock data for demonstration
      setPredictions([
        {
          symbol: 'AAPL',
          timeframe: '1d',
          direction: 'bullish',
          confidence: 0.82,
          targetPrice: 185.50,
          stopLoss: 175.00,
          reasoning: 'Strong technical momentum with positive sentiment indicators',
          technicalIndicators: {
            rsi: 65,
            macd: { signal: 0.5, histogram: 0.3 },
            bollinger: { upper: 182, lower: 175, middle: 178.5 },
            support: [175, 172],
            resistance: [185, 190],
          },
          fundamentalFactors: ['Strong earnings', 'Product innovation', 'Market expansion'],
          riskScore: 0.25,
          expectedReturn: 0.025,
          timeHorizon: '1-2 weeks',
        },
        {
          symbol: 'MSFT',
          timeframe: '1d',
          direction: 'bearish',
          confidence: 0.71,
          targetPrice: 365.00,
          stopLoss: 375.00,
          reasoning: 'Technical indicators showing weakness with market uncertainty',
          technicalIndicators: {
            rsi: 45,
            macd: { signal: -0.3, histogram: -0.2 },
            bollinger: { upper: 375, lower: 360, middle: 367.5 },
            support: [360, 355],
            resistance: [375, 380],
          },
          fundamentalFactors: ['Market volatility', 'Sector rotation'],
          riskScore: 0.35,
          expectedReturn: -0.015,
          timeHorizon: '1-2 weeks',
        },
      ]);
    }
  };

  const loadSignals = async () => {
    try {
      const signalsData = await aiService.getActiveSignals();
      setSignals(signalsData);
    } catch (error) {
      console.error('Failed to load signals:', error);
      // Mock data for demonstration
      const mockSignals: TradingSignal[] = [];
      const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
      const actions = ['buy', 'sell', 'hold'];
      const strengths = [0.3, 0.6, 0.9]; // Changed to numbers
      
      for (let i = 0; i < 20; i++) {
        mockSignals.push({
          id: `signal-${i}`,
          strategyId: `strategy-${Math.floor(Math.random() * 3)}`,
          symbol: symbols[Math.floor(Math.random() * symbols.length)],
          action: actions[Math.floor(Math.random() * actions.length)] as 'buy' | 'sell' | 'hold',
          strength: strengths[Math.floor(Math.random() * strengths.length)],
          confidence: 0.5 + Math.random() * 0.4,
          price: 100 + Math.random() * 200,
          timestamp: new Date(Date.now() - Math.random() * 3600000),
          reasoning: 'AI model detected strong momentum pattern with high probability of continuation',
          technicalAnalysis: {
            indicators: {
              rsi: 50 + Math.random() * 40,
              macd: Math.random() * 2 - 1,
              sma: 100 + Math.random() * 50
            },
            patterns: ['bullish_flag', 'ascending_triangle'],
            support: 90 + Math.random() * 20,
            resistance: 120 + Math.random() * 30
          },
          riskMetrics: {
            volatility: 0.1 + Math.random() * 0.3,
            beta: 0.8 + Math.random() * 0.4,
            var: 0.05 + Math.random() * 0.1,
            expectedShortfall: 0.08 + Math.random() * 0.12
          }
        });
      }
      setSignals(mockSignals);
    }
  };

  const loadInsights = async () => {
    try {
      const insightsData = await aiService.getAIInsights();
      setInsights(insightsData);
    } catch (error) {
      console.error('Failed to load insights:', error);
      // Mock data for demonstration
      setInsights([
        {
          id: 'insight-1',
          type: 'market_outlook',
          title: 'Market Regime Change Detected',
          description: 'AI models indicate a shift from low to high volatility regime with 85% confidence',
          confidence: 0.85,
          impact: 'high',
          actionable: true,
          recommendations: [
            'Reduce position sizes to manage increased volatility',
            'Consider hedging strategies for downside protection',
            'Monitor correlation changes between assets',
          ],
          affectedAssets: ['SPY', 'QQQ', 'AAPL', 'MSFT'],
          timeframe: '1-2 weeks',
          createdAt: new Date(),
          metadata: {}
        },
        {
          id: 'insight-2',
          type: 'opportunity',
          title: 'Sentiment Divergence in Tech Sector',
          description: 'News sentiment turning negative while price action remains positive - potential reversal signal',
          confidence: 0.72,
          impact: 'medium',
          actionable: true,
          recommendations: [
            'Monitor tech sector positions closely',
            'Consider taking profits on overextended positions',
            'Watch for confirmation in volume patterns',
          ],
          affectedAssets: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
          timeframe: '3-5 days',
          createdAt: new Date(),
          metadata: {}
        },
      ]);
    }
  };

  const loadModelMetrics = async () => {
    try {
      // Mock data for demonstration
      setModelMetrics({
        accuracy: 0.74,
        precision: 0.71,
        recall: 0.68,
        f1Score: 0.69,
        auc: 0.78,
        sharpeRatio: 1.45,
        maxDrawdown: 0.08,
        winRate: 0.68,
        avgReturn: 0.125,
        volatility: 0.18,
        lastUpdated: new Date(),
      });
    } catch (error) {
      console.error('Failed to load model metrics:', error);
    }
  };

  const handleStrategyAction = async (strategyId: string, action: 'start' | 'pause' | 'stop') => {
    try {
      // Mock implementation - in real app this would call aiService.controlStrategy
      console.log(`${action} strategy ${strategyId}`);
      
      // Update strategy status
      setStrategies(prev => prev.map(s => 
        s.id === strategyId 
          ? { ...s, status: action === 'start' ? 'active' : action === 'pause' ? 'paused' : 'stopped' }
          : s
      ));
    } catch (error) {
      console.error(`Failed to ${action} strategy:`, error);
    }
  };

  const handleConfigureStrategy = (strategyId: string) => {
    // TODO: Open strategy configuration modal or navigate to configuration page
    console.log('Configure strategy:', strategyId);
    // For now, just log the action - this could open a modal or navigate to a config page
    alert(`Configuration for strategy ${strategyId} - Feature coming soon!`);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      case 'stopped': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getSignalColor = (action: string) => {
    switch (action) {
      case 'buy': return 'text-green-600';
      case 'sell': return 'text-red-600';
      case 'hold': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getStrengthColor = (strength: number | string) => {
    let strengthLevel: string;
    
    if (typeof strength === 'number') {
      if (strength >= 0.7) strengthLevel = 'strong';
      else if (strength >= 0.4) strengthLevel = 'moderate';
      else strengthLevel = 'weak';
    } else {
      strengthLevel = strength;
    }
    
    switch (strengthLevel) {
      case 'strong': return 'bg-green-100 text-green-800';
      case 'moderate': return 'bg-yellow-100 text-yellow-800';
      case 'weak': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const renderStrategies = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {strategies.map((strategy) => (
          <Card key={strategy.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">{strategy.name}</CardTitle>
                  <p className="text-sm text-gray-600 mt-1">{strategy.description}</p>
                </div>
                <Badge className={getStatusColor(strategy.status || 'stopped')}>
                  {strategy.status || 'stopped'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Performance Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Expected Return</p>
                    <p className="text-lg font-semibold text-green-600">
                      +{(strategy.expectedReturn * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Sharpe Ratio</p>
                    <p className="text-lg font-semibold">{strategy.sharpeRatio.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Win Rate</p>
                    <p className="text-lg font-semibold">{(strategy.winRate * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Max Drawdown</p>
                    <p className="text-lg font-semibold text-red-600">
                      -{(strategy.maxDrawdown * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Win Rate</span>
                    <span>{(strategy.winRate * 100).toFixed(0)}%</span>
                  </div>
                  <Progress value={strategy.winRate * 100} className="h-2" />
                </div>

                {/* Action Buttons */}
                <div className="flex space-x-2">
                  {strategy.status !== 'active' && (
                    <Button
                      size="sm"
                      onClick={() => handleStrategyAction(strategy.id, 'start')}
                      className="flex items-center space-x-1"
                    >
                      <PlayIcon className="h-4 w-4" />
                      <span>Start</span>
                    </Button>
                  )}
                  {strategy.status === 'active' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleStrategyAction(strategy.id, 'pause')}
                      className="flex items-center space-x-1"
                    >
                      <PauseIcon className="h-4 w-4" />
                      <span>Pause</span>
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleStrategyAction(strategy.id, 'stop')}
                    className="flex items-center space-x-1"
                  >
                    <StopIcon className="h-4 w-4" />
                    <span>Stop</span>
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => handleConfigureStrategy(strategy.id)}
                    className="flex items-center space-x-1"
                  >
                    <CogIcon className="h-4 w-4" />
                    <span>Configure</span>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Model Performance Metrics */}
      {modelMetrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <ChartBarIcon className="h-5 w-5" />
              <span>Model Performance Metrics</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold">{(modelMetrics.accuracy * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-600">Accuracy</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{(modelMetrics.precision * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-600">Precision</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{(modelMetrics.recall * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-600">Recall</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{modelMetrics.sharpeRatio.toFixed(2)}</p>
                <p className="text-sm text-gray-600">Sharpe Ratio</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-red-600">-{(modelMetrics.maxDrawdown * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-600">Max Drawdown</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderPredictions = () => (
    <div className="space-y-6">
      {predictions.map((prediction) => (
        <Card key={prediction.symbol}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-xl">{prediction.symbol}</CardTitle>
              <div className="flex items-center space-x-2">
                <Badge className={prediction.direction === 'bullish' ? 'bg-green-100 text-green-800' : prediction.direction === 'bearish' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'}>
                  {prediction.direction.toUpperCase()}
                </Badge>
                <span className="text-sm text-gray-600">
                  {(prediction.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Prediction Details */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Target Price</p>
                  <p className={`text-lg font-semibold ${
                    prediction.direction === 'bullish' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ${prediction.targetPrice.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Expected Return</p>
                  <p className="text-lg font-semibold">
                    {(prediction.expectedReturn * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Risk Score</p>
                  <p className="text-lg font-semibold">
                    {prediction.riskScore.toFixed(1)}/10
                  </p>
                </div>
              </div>

              {/* Contributing Factors */}
              <div>
                <h4 className="font-semibold mb-3">Fundamental Factors</h4>
                <div className="space-y-2">
                  {prediction.fundamentalFactors.map((factor, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm">{factor}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm">{factor}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderSignals = () => (
    <div className="space-y-4">
      {signals.map((signal) => (
        <Card key={signal.id} className="hover:shadow-md transition-shadow">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div>
                  <h4 className="font-semibold">{signal.symbol}</h4>
                  <p className="text-sm text-gray-600">{new Date(signal.timestamp).toLocaleTimeString()}</p>
                </div>
                <div className={`font-semibold ${getSignalColor(signal.action)}`}>
                  {signal.action.toUpperCase()}
                </div>
                <Badge className={getStrengthColor(signal.strength)}>
                  {signal.strength}
                </Badge>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-600">Confidence</p>
                <p className="font-semibold">{(signal.confidence * 100).toFixed(0)}%</p>
              </div>
            </div>
            
            <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Current Price</p>
                <p className="font-medium">${signal.price.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-600">Target Price</p>
                <p className="font-medium text-green-600">${(signal.price * 1.05).toFixed(2)}</p>
              </div>
              <div>
                <p className="text-gray-600">Stop Loss</p>
                <p className="font-medium text-red-600">${(signal.price * 0.95).toFixed(2)}</p>
              </div>
            </div>
            
            <div className="mt-3">
              <p className="text-sm text-gray-700">{signal.reasoning}</p>
              <p className="text-xs text-gray-500 mt-1">
                {signal.timestamp.toLocaleString()}
              </p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderInsights = () => (
    <div className="space-y-4">
      {insights.map((insight) => (
        <Card key={insight.id}>
          <CardContent className="p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {insight.type === 'market_outlook' && <ChartBarIcon className="h-6 w-6 text-blue-600" />}
                {insight.type === 'opportunity' && <LightBulbIcon className="h-6 w-6 text-yellow-600" />}
                {insight.type === 'risk_alert' && <ExclamationTriangleIcon className="h-6 w-6 text-red-600" />}
                {insight.type === 'portfolio_optimization' && <CheckCircleIcon className="h-6 w-6 text-green-600" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold">{insight.title}</h4>
                  <div className="flex items-center space-x-2">
                    <Badge className={insight.impact === 'high' ? 'bg-red-100 text-red-800' : 
                                   insight.impact === 'medium' ? 'bg-yellow-100 text-yellow-800' : 
                                   'bg-green-100 text-green-800'}>
                      {insight.impact} impact
                    </Badge>
                    <span className="text-sm text-gray-600">
                      {(insight.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
                
                <p className="text-gray-700 mb-3">{insight.description}</p>
                
                {insight.actionable && insight.recommendations && (
                  <div>
                    <h5 className="font-medium mb-2">Recommendations:</h5>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                      {insight.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <p className="text-xs text-gray-500 mt-3">
                  {insight.createdAt.toLocaleString()}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">AI Trading Strategies</h2>
        <div className="flex items-center space-x-2">
          <ClockIcon className="h-4 w-4 text-gray-500" />
          <span className="text-sm text-gray-600">
            Last updated: {new Date().toLocaleTimeString()}
          </span>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="strategies">Strategies</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="signals">Signals</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="strategies" className="space-y-6">
          {renderStrategies()}
        </TabsContent>

        <TabsContent value="predictions" className="space-y-6">
          {renderPredictions()}
        </TabsContent>

        <TabsContent value="signals" className="space-y-6">
          {renderSignals()}
        </TabsContent>

        <TabsContent value="insights" className="space-y-6">
          {renderInsights()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AITradingStrategies;