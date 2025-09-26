'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  BanknotesIcon,
  BeakerIcon,
  BoltIcon,
  CalculatorIcon,
  ChartBarIcon,
  ChartPieIcon,
  ClockIcon,
  Cog6ToothIcon,
  CpuChipIcon,
  CurrencyDollarIcon,
  EyeIcon,
  EyeSlashIcon,
  GlobeAltIcon,
  InformationCircleIcon,
  LightBulbIcon,
  ScaleIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';

interface TechnicalIndicator {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters?: Record<string, any>;
  enabled: boolean;
  assetTypes: string[];
  complexity: 'basic' | 'intermediate' | 'advanced';
  icon: React.ComponentType<any>;
}

interface TechnicalsInterfaceProps {
  selectedAssetType: 'crypto' | 'stock' | 'forex' | 'futures' | 'index';
  onIndicatorToggle: (indicatorId: string, enabled: boolean, parameters?: any) => void;
  enabledIndicators: string[];
  onAssetTypeChange: (assetType: string) => void;
}

const TechnicalsInterface: React.FC<TechnicalsInterfaceProps> = ({
  selectedAssetType,
  onIndicatorToggle,
  enabledIndicators,
  onAssetTypeChange
}) => {
  const [activeTab, setActiveTab] = useState(selectedAssetType);
  const [searchTerm, setSearchTerm] = useState('');
  const [complexityFilter, setComplexityFilter] = useState<string>('all');
  const [expandedIndicator, setExpandedIndicator] = useState<string | null>(null);

  // Comprehensive indicator definitions
  const indicators: TechnicalIndicator[] = [
    // ==================== CRYPTO INDICATORS ====================
    {
      id: 'stock_to_flow',
      name: 'Stock-to-Flow (S2F)',
      category: 'Fundamental',
      description: 'Scarcity-based valuation model measuring the ratio of existing supply to new production',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: ScaleIcon,
      parameters: { halvingCycle: 4, modelType: 'bitcoin' }
    },
    {
      id: 'metcalfes_law',
      name: "Metcalfe's Law",
      category: 'Network',
      description: 'Network value proportional to the square of connected users',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: GlobeAltIcon,
      parameters: { networkMetric: 'active_addresses' }
    },
    {
      id: 'nvt_ratio',
      name: 'NVT Ratio',
      category: 'Valuation',
      description: 'Network Value to Transactions ratio - crypto PE ratio equivalent',
      assetTypes: ['crypto'],
      complexity: 'intermediate',
      enabled: false,
      icon: CalculatorIcon,
      parameters: { period: 30, smoothing: true }
    },
    {
      id: 'nvm_ratio',
      name: 'NVM Ratio',
      category: 'Valuation',
      description: 'Network Value to Metcalfe ratio for network efficiency analysis',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartPieIcon
    },
    {
      id: 'crypto_quant_metrics',
      name: 'Crypto Quant Metrics',
      category: 'On-Chain',
      description: 'Advanced on-chain metrics including MVRV, SOPR, and exchange flows',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: BeakerIcon,
      parameters: { metrics: ['mvrv', 'sopr', 'exchange_flow'] }
    },
    {
      id: 'log_regression',
      name: 'Logarithmic Regression',
      category: 'Technical',
      description: 'Long-term price channel analysis using logarithmic regression bands',
      assetTypes: ['crypto'],
      complexity: 'intermediate',
      enabled: false,
      icon: ArrowTrendingUpIcon,
      parameters: { degree: 2, bands: 2 }
    },
    {
      id: 'finbert_crypto',
      name: 'FinBERT/CryptoBERT',
      category: 'Sentiment',
      description: 'AI-powered sentiment analysis using specialized BERT models',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: CpuChipIcon,
      parameters: { model: 'cryptobert', sources: ['twitter', 'reddit', 'news'] }
    },
    {
      id: 'onchain_ml',
      name: 'On-chain ML Models',
      category: 'Machine Learning',
      description: 'Machine learning models trained on blockchain data',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: SparklesIcon,
      parameters: { model_type: 'ensemble', features: 'all' }
    },
    {
      id: 'gnn_analysis',
      name: 'Graph Neural Networks',
      category: 'Network Analysis',
      description: 'GNN-based analysis of blockchain transaction networks',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: GlobeAltIcon
    },
    {
      id: 'crypto_rl',
      name: 'Reinforcement Learning',
      category: 'AI Trading',
      description: 'RL agents trained on crypto market dynamics',
      assetTypes: ['crypto'],
      complexity: 'advanced',
      enabled: false,
      icon: BoltIcon,
      parameters: { agent: 'ppo', environment: 'crypto_trading' }
    },

    // ==================== STOCK INDICATORS ====================
    {
      id: 'dcf_model',
      name: 'Discounted Cash Flow (DCF)',
      category: 'Fundamental',
      description: 'Intrinsic value calculation based on projected cash flows',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: BanknotesIcon,
      parameters: { growth_rate: 0.05, discount_rate: 0.10, terminal_growth: 0.02 }
    },
    {
      id: 'ddm_model',
      name: 'Dividend Discount Model (DDM)',
      category: 'Fundamental',
      description: 'Valuation based on present value of expected dividends',
      assetTypes: ['stock'],
      complexity: 'intermediate',
      enabled: false,
      icon: CurrencyDollarIcon,
      parameters: { growth_rate: 0.03, required_return: 0.08 }
    },
    {
      id: 'capm_analysis',
      name: 'CAPM Analysis',
      category: 'Risk',
      description: 'Capital Asset Pricing Model for risk-adjusted returns',
      assetTypes: ['stock'],
      complexity: 'intermediate',
      enabled: false,
      icon: ScaleIcon,
      parameters: { risk_free_rate: 0.02, period: 252 }
    },
    {
      id: 'fama_french',
      name: 'Fama-French Model',
      category: 'Factor Analysis',
      description: 'Three-factor model including size and value factors',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartBarIcon,
      parameters: { factors: ['market', 'size', 'value'] }
    },
    {
      id: 'gordon_growth',
      name: 'Gordon Growth Model',
      category: 'Valuation',
      description: 'Dividend growth model for mature companies',
      assetTypes: ['stock'],
      complexity: 'basic',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'arima_stock',
      name: 'ARIMA Model',
      category: 'Time Series',
      description: 'Autoregressive Integrated Moving Average for price forecasting',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: ClockIcon,
      parameters: { order: [1, 1, 1], seasonal: false }
    },
    {
      id: 'garch_stock',
      name: 'GARCH Model',
      category: 'Volatility',
      description: 'Generalized Autoregressive Conditional Heteroskedasticity',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: ArrowTrendingUpIcon,
      parameters: { p: 1, q: 1 }
    },
    {
      id: 'var_model',
      name: 'VAR Model',
      category: 'Multivariate',
      description: 'Vector Autoregression for multiple time series analysis',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartBarIcon
    },
    {
      id: 'kalman_filter',
      name: 'Kalman Filters',
      category: 'State Space',
      description: 'Dynamic state estimation for trend and noise separation',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: BeakerIcon
    },
    {
      id: 'lstm_stock',
      name: 'LSTM Networks',
      category: 'Deep Learning',
      description: 'Long Short-Term Memory networks for sequence prediction',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: CpuChipIcon,
      parameters: { lookback: 60, layers: 2, units: 50 }
    },
    {
      id: 'xgboost_stock',
      name: 'XGBoost',
      category: 'Machine Learning',
      description: 'Gradient boosting for feature-based price prediction',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: SparklesIcon,
      parameters: { n_estimators: 100, max_depth: 6 }
    },
    {
      id: 'bayesian_nn',
      name: 'Bayesian Neural Networks',
      category: 'Probabilistic ML',
      description: 'Uncertainty-aware neural networks for risk assessment',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: LightBulbIcon
    },
    {
      id: 'automl_stock',
      name: 'AutoML',
      category: 'Automated ML',
      description: 'Automated machine learning model selection and tuning',
      assetTypes: ['stock'],
      complexity: 'advanced',
      enabled: false,
      icon: BoltIcon
    },

    // ==================== FOREX INDICATORS ====================
    {
      id: 'ppp_model',
      name: 'Purchasing Power Parity (PPP)',
      category: 'Fundamental',
      description: 'Exchange rate theory based on price level differences',
      assetTypes: ['forex'],
      complexity: 'intermediate',
      enabled: false,
      icon: ScaleIcon,
      parameters: { base_inflation: 0.02, quote_inflation: 0.025 }
    },
    {
      id: 'irp_model',
      name: 'Interest Rate Parity (IRP)',
      category: 'Arbitrage',
      description: 'No-arbitrage condition relating interest rates and exchange rates',
      assetTypes: ['forex'],
      complexity: 'intermediate',
      enabled: false,
      icon: BanknotesIcon
    },
    {
      id: 'uip_model',
      name: 'Uncovered Interest Parity (UIP)',
      category: 'Expectation',
      description: 'Exchange rate expectations based on interest rate differentials',
      assetTypes: ['forex'],
      complexity: 'advanced',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'balance_payments',
      name: 'Balance of Payments',
      category: 'Macroeconomic',
      description: 'Current account and capital flow analysis',
      assetTypes: ['forex'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartPieIcon
    },
    {
      id: 'monetary_models',
      name: 'Monetary Models',
      category: 'Central Bank',
      description: 'Exchange rate models based on monetary policy',
      assetTypes: ['forex'],
      complexity: 'advanced',
      enabled: false,
      icon: BanknotesIcon
    },
    {
      id: 'forex_garch',
      name: 'GARCH/EGARCH',
      category: 'Volatility',
      description: 'Volatility modeling for forex pairs',
      assetTypes: ['forex'],
      complexity: 'advanced',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'forex_rl',
      name: 'RL Agents',
      category: 'AI Trading',
      description: 'Reinforcement learning for forex trading strategies',
      assetTypes: ['forex'],
      complexity: 'advanced',
      enabled: false,
      icon: BoltIcon
    },
    {
      id: 'forexbert',
      name: 'ForexBERT Sentiment',
      category: 'Sentiment',
      description: 'Specialized BERT model for forex sentiment analysis',
      assetTypes: ['forex'],
      complexity: 'advanced',
      enabled: false,
      icon: CpuChipIcon
    },

    // ==================== FUTURES INDICATORS ====================
    {
      id: 'cost_of_carry',
      name: 'Cost-of-Carry Model',
      category: 'Pricing',
      description: 'Theoretical futures pricing based on storage and financing costs',
      assetTypes: ['futures'],
      complexity: 'intermediate',
      enabled: false,
      icon: CalculatorIcon,
      parameters: { risk_free_rate: 0.02, storage_cost: 0.01 }
    },
    {
      id: 'convenience_yield',
      name: 'Convenience Yield',
      category: 'Storage',
      description: 'Benefit of holding physical commodity vs futures',
      assetTypes: ['futures'],
      complexity: 'advanced',
      enabled: false,
      icon: BanknotesIcon
    },
    {
      id: 'samuelson_effect',
      name: 'Samuelson Effect',
      category: 'Maturity',
      description: 'Volatility increase as futures approach expiration',
      assetTypes: ['futures'],
      complexity: 'intermediate',
      enabled: false,
      icon: ClockIcon
    },
    {
      id: 'backwardation_contango',
      name: 'Backwardation/Contango',
      category: 'Term Structure',
      description: 'Analysis of futures curve shape and market conditions',
      assetTypes: ['futures'],
      complexity: 'intermediate',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'futures_var',
      name: 'VAR Models',
      category: 'Risk',
      description: 'Value at Risk calculation for futures positions',
      assetTypes: ['futures'],
      complexity: 'advanced',
      enabled: false,
      icon: ScaleIcon
    },
    {
      id: 'seasonal_arima',
      name: 'Seasonal ARIMA',
      category: 'Seasonality',
      description: 'Time series modeling with seasonal patterns',
      assetTypes: ['futures'],
      complexity: 'advanced',
      enabled: false,
      icon: ClockIcon
    },
    {
      id: 'momentum_mean_reversion',
      name: 'Momentum + Mean Reversion',
      category: 'Strategy',
      description: 'Combined momentum and mean reversion signals',
      assetTypes: ['futures'],
      complexity: 'intermediate',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'futures_rl',
      name: 'RL (PPO, SAC, DDPG)',
      category: 'AI Trading',
      description: 'Advanced RL algorithms for futures trading',
      assetTypes: ['futures'],
      complexity: 'advanced',
      enabled: false,
      icon: BoltIcon,
      parameters: { algorithm: 'ppo', environment: 'futures_trading' }
    },

    // ==================== INDEX INDICATORS ====================
    {
      id: 'macro_factors',
      name: 'Macroeconomic Factors',
      category: 'Fundamental',
      description: 'GDP, inflation, employment impact on index performance',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartPieIcon
    },
    {
      id: 'apt_model',
      name: 'Arbitrage Pricing Theory',
      category: 'Factor Model',
      description: 'Multi-factor model for asset pricing',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: ScaleIcon
    },
    {
      id: 'term_structure',
      name: 'Term Structure Models',
      category: 'Interest Rates',
      description: 'Yield curve analysis and bond pricing models',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'cointegration_vecm',
      name: 'Cointegration & VECM',
      category: 'Econometric',
      description: 'Long-run equilibrium relationships between series',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartBarIcon
    },
    {
      id: 'transformers_index',
      name: 'Transformers',
      category: 'Deep Learning',
      description: 'Attention-based models for sequence prediction',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: CpuChipIcon
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      category: 'Ensemble',
      description: 'Tree-based ensemble learning for classification/regression',
      assetTypes: ['index'],
      complexity: 'intermediate',
      enabled: false,
      icon: SparklesIcon
    },
    {
      id: 'sentiment_news',
      name: 'Sentiment & News Analysis',
      category: 'Alternative Data',
      description: 'News sentiment impact on index movements',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: LightBulbIcon
    },
    {
      id: 'hybrid_models',
      name: 'Hybrid Models (ARIMA-LSTM)',
      category: 'Combined',
      description: 'Combination of statistical and ML approaches',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: BeakerIcon
    },
    {
      id: 'elliott_fibonacci',
      name: 'Elliott Wave & Fibonacci',
      category: 'Technical',
      description: 'Wave pattern analysis with Fibonacci retracements',
      assetTypes: ['index'],
      complexity: 'intermediate',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'markowitz_factor',
      name: 'Markowitz & Factor Investing',
      category: 'Portfolio',
      description: 'Modern portfolio theory with factor tilts',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: ChartPieIcon
    },
    {
      id: 'volatility_regime',
      name: 'Volatility Regime Switching',
      category: 'Regime',
      description: 'Hidden Markov models for volatility regimes',
      assetTypes: ['index'],
      complexity: 'advanced',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },

    // ==================== CROSS-ASSET INDICATORS ====================
    {
      id: 'cross_arima',
      name: 'ARIMA/SARIMA',
      category: 'Time Series',
      description: 'Cross-asset time series analysis',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: ClockIcon
    },
    {
      id: 'cross_garch',
      name: 'GARCH Models',
      category: 'Volatility',
      description: 'Cross-asset volatility modeling',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: ArrowTrendingUpIcon
    },
    {
      id: 'cross_lstm',
      name: 'LSTM/GRU',
      category: 'Deep Learning',
      description: 'Recurrent neural networks for all asset classes',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: CpuChipIcon
    },
    {
      id: 'cross_transformer',
      name: 'Transformer Models',
      category: 'Attention',
      description: 'Attention-based models for cross-asset analysis',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: SparklesIcon
    },
    {
      id: 'cross_xgboost',
      name: 'XGBoost/LightGBM',
      category: 'Gradient Boosting',
      description: 'Advanced gradient boosting for all assets',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: BoltIcon
    },
    {
      id: 'cross_svm',
      name: 'Support Vector Machines',
      category: 'Classification',
      description: 'SVM for pattern recognition across assets',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'intermediate',
      enabled: false,
      icon: BeakerIcon
    },
    {
      id: 'cross_rsi_macd',
      name: 'RSI/MACD/Ichimoku',
      category: 'Technical',
      description: 'Classic technical indicators for all assets',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'basic',
      enabled: false,
      icon: ChartBarIcon
    },
    {
      id: 'cross_rl',
      name: 'RL (PPO, SAC, DDPG)',
      category: 'Reinforcement Learning',
      description: 'Multi-asset reinforcement learning agents',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: BoltIcon
    },
    {
      id: 'markowitz_mpt',
      name: 'Markowitz MPT',
      category: 'Portfolio',
      description: 'Modern Portfolio Theory optimization',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'intermediate',
      enabled: false,
      icon: ChartPieIcon
    },
    {
      id: 'monte_carlo',
      name: 'Monte Carlo',
      category: 'Simulation',
      description: 'Monte Carlo simulation for risk assessment',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: CalculatorIcon
    },
    {
      id: 'multi_bert',
      name: 'Multi-BERT Models',
      category: 'NLP',
      description: 'FinBERT, CryptoBERT, ForexBERT ensemble',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: CpuChipIcon
    },
    {
      id: 'hmm_bayesian',
      name: 'HMM & Bayesian Change Point',
      category: 'State Space',
      description: 'Hidden Markov Models and change point detection',
      assetTypes: ['crypto', 'stock', 'forex', 'futures', 'index'],
      complexity: 'advanced',
      enabled: false,
      icon: LightBulbIcon
    }
  ];

  // Filter indicators based on selected asset type, search term, and complexity
  const filteredIndicators = indicators.filter(indicator => {
    const matchesAssetType = indicator.assetTypes.includes(activeTab);
    const matchesSearch = indicator.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         indicator.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         indicator.category.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesComplexity = complexityFilter === 'all' || indicator.complexity === complexityFilter;
    
    return matchesAssetType && matchesSearch && matchesComplexity;
  });

  // Group indicators by category
  const groupedIndicators = filteredIndicators.reduce((groups, indicator) => {
    const category = indicator.category;
    if (!groups[category]) {
      groups[category] = [];
    }
    groups[category].push(indicator);
    return groups;
  }, {} as Record<string, TechnicalIndicator[]>);

  const handleIndicatorToggle = (indicator: TechnicalIndicator) => {
    const isEnabled = enabledIndicators.includes(indicator.id);
    onIndicatorToggle(indicator.id, !isEnabled, indicator.parameters);
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'basic': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const assetTypeIcons = {
    crypto: CurrencyDollarIcon,
    stock: ArrowTrendingUpIcon,
    forex: GlobeAltIcon,
    futures: ChartBarIcon,
    index: ChartPieIcon
  };

  useEffect(() => {
    setActiveTab(selectedAssetType);
  }, [selectedAssetType]);

  return (
    <Card className="w-full mb-6">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <BeakerIcon className="h-6 w-6" />
            <span>Technical Indicators</span>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Badge variant="secondary">
              {filteredIndicators.length} indicators
            </Badge>
            <Badge variant="outline">
              {enabledIndicators.length} active
            </Badge>
          </div>
        </div>
        
        {/* Search and Filter Controls */}
        <div className="flex flex-wrap gap-4 mt-4">
          <div className="flex-1 min-w-64">
            <Input
              placeholder="Search indicators..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full"
            />
          </div>
          <Select value={complexityFilter} onValueChange={setComplexityFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Complexity" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="basic">Basic</SelectItem>
              <SelectItem value="intermediate">Intermediate</SelectItem>
              <SelectItem value="advanced">Advanced</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value) => {
          setActiveTab(value as any);
          onAssetTypeChange(value);
        }}>
          <TabsList className="grid w-full grid-cols-5">
            {Object.entries(assetTypeIcons).map(([type, Icon]) => (
              <TabsTrigger key={type} value={type} className="flex items-center space-x-2">
                <Icon className="h-4 w-4" />
                <span className="capitalize">{type}</span>
              </TabsTrigger>
            ))}
          </TabsList>
          
          {Object.keys(assetTypeIcons).map(assetType => (
            <TabsContent key={assetType} value={assetType} className="mt-6">
              <div className="space-y-6">
                {Object.entries(groupedIndicators).map(([category, categoryIndicators]) => (
                  <div key={category} className="space-y-3">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">
                      {category}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {categoryIndicators.map(indicator => {
                        const Icon = indicator.icon;
                        const isEnabled = enabledIndicators.includes(indicator.id);
                        const isExpanded = expandedIndicator === indicator.id;
                        
                        return (
                          <div
                            key={indicator.id}
                            className={`p-4 border rounded-lg transition-all duration-200 hover:shadow-md ${
                              isEnabled 
                                ? 'border-blue-300 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-700' 
                                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                            }`}
                          >
                            <div className="flex items-start justify-between mb-2">
                              <div className="flex items-center space-x-2">
                                <Icon className={`h-5 w-5 ${
                                  isEnabled ? 'text-blue-600 dark:text-blue-400' : 'text-gray-500'
                                }`} />
                                <h4 className="font-medium text-sm">{indicator.name}</h4>
                              </div>
                              <div className="flex items-center space-x-1">
                                <Badge 
                                  variant="secondary" 
                                  className={`text-xs ${getComplexityColor(indicator.complexity)}`}
                                >
                                  {indicator.complexity}
                                </Badge>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => setExpandedIndicator(isExpanded ? null : indicator.id)}
                                  className="h-6 w-6 p-0"
                                >
                                  <InformationCircleIcon className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                            
                            <p className="text-xs text-gray-600 dark:text-gray-400 mb-3 line-clamp-2">
                              {indicator.description}
                            </p>
                            
                            {isExpanded && indicator.parameters && (
                              <div className="mb-3 p-2 bg-gray-50 dark:bg-gray-800 rounded text-xs">
                                <strong>Parameters:</strong>
                                <pre className="mt-1 text-xs">
                                  {JSON.stringify(indicator.parameters, null, 2)}
                                </pre>
                              </div>
                            )}
                            
                            <div className="flex items-center justify-between">
                              <Button
                                variant={isEnabled ? "default" : "outline"}
                                size="sm"
                                onClick={() => handleIndicatorToggle(indicator)}
                                className="flex items-center space-x-1"
                              >
                                {isEnabled ? (
                                  <>
                                    <EyeSlashIcon className="h-3 w-3" />
                                    <span>Disable</span>
                                  </>
                                ) : (
                                  <>
                                    <EyeIcon className="h-3 w-3" />
                                    <span>Enable</span>
                                  </>
                                )}
                              </Button>
                              
                              {isEnabled && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0"
                                >
                                  <Cog6ToothIcon className="h-4 w-4" />
                                </Button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                
                {Object.keys(groupedIndicators).length === 0 && (
                  <div className="text-center py-12">
                    <BeakerIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      No indicators found
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">
                      Try adjusting your search terms or complexity filter.
                    </p>
                  </div>
                )}
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default TechnicalsInterface;