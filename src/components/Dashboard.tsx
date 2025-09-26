'use client';

import React, { useEffect, useState } from 'react';
import { useStore, Asset } from '../store/useStore';
import { assetsAPI, aiAPI } from '../services/api';
import webSocketService from '../services/websocket';
import TradingChart from './TradingChart';
import ModelSelector from './ModelSelector';
import Watchlist from './Watchlist';
import AIInsights from './AIInsights';
import MarketOverview from './MarketOverview';
import LoadingSpinner from './ui/LoadingSpinner';
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
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  BanknotesIcon,
  UserGroupIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  BellIcon,
  CogIcon,
  EyeIcon,
  PlusIcon,
  BuildingOfficeIcon,
  ShieldCheckIcon,
  RocketLaunchIcon,
  HeartIcon,
} from '@heroicons/react/24/outline';
import AITradingStrategies from './AITradingStrategies';
import SocialTradingDashboard from './SocialTradingDashboard';
import InstitutionalToolsDashboard from './InstitutionalToolsDashboard';
import RegulatoryComplianceDashboard from './RegulatoryComplianceDashboard';
import Subscription from './subscription';
import FeatureGate from './featuregate';
import TrialBanner from './trialbanner';
import PortfolioManager from './PortfolioManager';
import TradingInterface from './TradingInterface';
import AnalyticsDashboard from './AnalyticsDashboard';
import CommunityForum from './CommunityForum';
import NotificationCenter from './NotificationCenter';
import ChartAnalysis from './ChartAnalysis';
import PipelineControlPanel from './PipelineControlPanel';

const Dashboard: React.FC = () => {
  const {
    selectedAssets,
    selectedModels,
    chartData,
    isLoading,
    error,
    setAssets,
    setChartData,
    setLoading,
    setError,
    addAIInsight
  } = useStore();
  
  const [isConnected, setIsConnected] = useState(false);
  const [assets, setAssetsData] = useState<Asset[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'portfolio' | 'trading' | 'analytics' | 'community' | 'notifications' | 'ai-strategies' | 'social-trading' | 'institutional' | 'compliance' | 'subscription' | 'pipeline'>('overview');

  // Initialize WebSocket connection
  useEffect(() => {
    const initWebSocket = async () => {
      try {
        await webSocketService.connect();
        setIsConnected(true);
        
        // Set up event listeners
        webSocketService.on('market_update', (data) => {
          setAssetsData(data.data);
          setAssets(data.data);
        });
        
        webSocketService.on('ai_insight', (data) => {
          addAIInsight(data.data);
        });
        
        webSocketService.on('connect', () => {
          setIsConnected(true);
        });
        
        webSocketService.on('disconnect', () => {
          setIsConnected(false);
        });
        
      } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        setError('Failed to connect to real-time data');
      }
    };
    
    initWebSocket();
    
    return () => {
      webSocketService.disconnect();
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      setLoading(true);
      try {
        // Fetch assets
        const assetsResponse = await assetsAPI.getAssets();
        setAssets(assetsResponse.data);
        setAssetsData(assetsResponse.data);
        
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
        setError('Failed to load market data');
      } finally {
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);

  // Subscribe to real-time data for selected assets
  useEffect(() => {
    if (selectedAssets.length > 0 && isConnected) {
      selectedAssets.forEach(asset => {
        webSocketService.subscribeToAsset(asset.symbol);
        
        // Fetch chart data
        assetsAPI.getAssetChart(asset.symbol, '1d', 100)
          .then(data => {
            setChartData(asset.symbol, data);
          })
          .catch(error => {
            console.error('Failed to fetch chart data:', error);
          });
      });
    }
  }, [selectedAssets, isConnected]);

  // Run AI analysis for selected assets
  useEffect(() => {
    if (selectedAssets.length > 0) {
      selectedAssets.forEach(asset => {
        // Run AI analysis
        aiAPI.getExplanation({
          asset_id: asset.symbol,
          question: 'Provide comprehensive analysis'
        })
        .then(analysis => {
          addAIInsight({
            id: `${asset.symbol}-${Date.now()}`,
            title: `Analysis for ${asset.name}`,
            content: analysis.explanation,
            confidence: analysis.confidence,
            timestamp: new Date().toISOString(),
            assetId: asset.symbol
          });
        })
        .catch(error => {
          console.error('Failed to get AI analysis:', error);
        });
      });
    }
  }, [selectedAssets]);

  if (isLoading && !chartData.length) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="text-red-500 mb-2">Error loading dashboard data</div>
          <div className="text-gray-500 text-sm">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Market Overview */}
      <MarketOverview />
      
      {/* Trial Banner */}
      <TrialBanner />
      
      {/* Connection Status */}
      {!isConnected && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-3"></div>
            <span className="text-yellow-800 dark:text-yellow-200 text-sm">
              Reconnecting to real-time data...
            </span>
          </div>
        </div>
      )}
      
      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)}>
        <TabsList className="grid w-full grid-cols-12">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
          <TabsTrigger value="trading">Trading</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="chart-analysis">
            <ChartBarIcon className="h-4 w-4 mr-1" />
            Chart Analysis
          </TabsTrigger>
          <TabsTrigger value="pipeline">
            <CogIcon className="h-4 w-4 mr-1" />
            Pipeline
          </TabsTrigger>
          <TabsTrigger value="community">Community</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="subscription">Subscription</TabsTrigger>
          <TabsTrigger value="ai-strategies">
            <RocketLaunchIcon className="h-4 w-4 mr-1" />
            AI Strategies
          </TabsTrigger>
          <TabsTrigger value="social-trading">
            <HeartIcon className="h-4 w-4 mr-1" />
            Social Trading
          </TabsTrigger>
          <TabsTrigger value="institutional">
            <BuildingOfficeIcon className="h-4 w-4 mr-1" />
            Institutional
          </TabsTrigger>
          <TabsTrigger value="compliance">
            <ShieldCheckIcon className="h-4 w-4 mr-1" />
            Compliance
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
           {/* Main Dashboard Grid */}
           <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
             {/* Left Column - Chart and Models */}
             <div className="lg:col-span-3 space-y-6">
               {/* Trading Chart */}
               <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                 <div className="p-6">
                   <div className="flex items-center justify-between mb-6">
                     <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                       Price Chart
                     </h2>
                     <div className="flex items-center space-x-2">
                       {selectedAssets.length > 0 ? (
                         <div className="flex items-center space-x-2">
                           <span className="text-sm text-gray-500 dark:text-gray-400">
                             {selectedAssets[0].symbol}
                           </span>
                           <span className={`text-sm font-medium ${
                             selectedAssets[0].change24h >= 0 
                               ? 'text-green-600 dark:text-green-400' 
                               : 'text-red-600 dark:text-red-400'
                           }`}>
                             {selectedAssets[0].change24h >= 0 ? '+' : ''}{selectedAssets[0].change24h?.toFixed(2)}%
                           </span>
                         </div>
                       ) : (
                         <span className="text-sm text-gray-500 dark:text-gray-400">
                           Select an asset from watchlist
                         </span>
                       )}
                       {isConnected && (
                         <div className="flex items-center space-x-1">
                           <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                           <span className="text-xs text-gray-500 dark:text-gray-400">Live</span>
                         </div>
                       )}
                     </div>
                   </div>
                   <TradingChart />
                 </div>
               </div>
 
               {/* Model Selector */}
               <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                 <div className="p-6">
                   <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                     AI Models & Analysis
                   </h2>
                   <ModelSelector />
                 </div>
               </div>
             </div>
 
             {/* Right Column - Watchlist and Insights */}
             <div className="space-y-6">
               {/* Watchlist */}
               <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                 <div className="p-6">
                   <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                     Watchlist
                   </h2>
                   <Watchlist />
                 </div>
               </div>
 
               {/* AI Insights */}
               <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                 <div className="p-6">
                   <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                     AI Insights
                   </h2>
                   <AIInsights />
                 </div>
               </div>
             </div>
           </div>
         </TabsContent>

         <TabsContent value="portfolio" className="space-y-6">
           <PortfolioManager />
         </TabsContent>

         <TabsContent value="trading" className="space-y-6">
           <TradingInterface />
         </TabsContent>

         <TabsContent value="analytics" className="space-y-6">
           <FeatureGate feature="advancedAnalytics">
             <AnalyticsDashboard />
           </FeatureGate>
         </TabsContent>

         <TabsContent value="chart-analysis" className="space-y-6">
           <FeatureGate feature="aiInsights">
             <ChartAnalysis />
           </FeatureGate>
         </TabsContent>

         <TabsContent value="community" className="space-y-6">
           <FeatureGate feature="socialTrading">
             <CommunityForum />
           </FeatureGate>
         </TabsContent>

         <TabsContent value="notifications" className="space-y-6">
           <NotificationCenter />
         </TabsContent>

        <TabsContent value="ai-strategies" className="space-y-6">
          <FeatureGate feature="aiInsights">
            <AITradingStrategies />
          </FeatureGate>
        </TabsContent>

        <TabsContent value="social-trading" className="space-y-6">
          <FeatureGate feature="socialTrading">
            <SocialTradingDashboard />
          </FeatureGate>
        </TabsContent>

        <TabsContent value="institutional" className="space-y-6">
          <FeatureGate feature="institutionalTools">
            <InstitutionalToolsDashboard />
          </FeatureGate>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-6">
          <FeatureGate feature="complianceTools">
            <RegulatoryComplianceDashboard />
          </FeatureGate>
        </TabsContent>

        <TabsContent value="pipeline" className="space-y-6">
          <PipelineControlPanel />
        </TabsContent>

        <TabsContent value="subscription" className="space-y-6">
          <Subscription />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Dashboard;