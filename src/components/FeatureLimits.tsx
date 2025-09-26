import React from 'react';
import { useFeatureAccess } from '../hooks/useFeatureAccess';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import {
  ChartBarIcon,
  EyeIcon,
  BellIcon,
  CloudIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import { Infinity } from 'lucide-react';

interface UsageData {
  portfolios: number;
  watchlistItems: number;
  alerts: number;
  apiCalls: number;
}

interface FeatureLimitsProps {
  usage: UsageData;
  className?: string;
}

const FeatureLimits: React.FC<FeatureLimitsProps> = ({ usage, className = '' }) => {
  const { featureLimits, checkLimit, subscription } = useFeatureAccess();
  
  const formatLimit = (limit: number): string => {
    return limit === -1 ? 'Unlimited' : limit.toString();
  };
  
  const getUsagePercentage = (current: number, limit: number): number => {
    if (limit === -1) return 0; // Unlimited
    return Math.min((current / limit) * 100, 100);
  };
  
  const getUsageColor = (percentage: number): string => {
    if (percentage >= 90) return 'bg-red-500';
    if (percentage >= 75) return 'bg-yellow-500';
    return 'bg-green-500';
  };
  
  const isNearLimit = (current: number, limit: number): boolean => {
    if (limit === -1) return false;
    return (current / limit) >= 0.8;
  };
  
  const limitItems = [
    {
      icon: ChartBarIcon,
      label: 'Portfolios',
      current: usage.portfolios,
      limit: featureLimits.maxPortfolios,
      key: 'maxPortfolios' as const,
    },
    {
      icon: EyeIcon,
      label: 'Watchlist Items',
      current: usage.watchlistItems,
      limit: featureLimits.maxWatchlistItems,
      key: 'maxWatchlistItems' as const,
    },
    {
      icon: BellIcon,
      label: 'Alerts',
      current: usage.alerts,
      limit: featureLimits.maxAlerts,
      key: 'maxAlerts' as const,
    },
    {
      icon: CloudIcon,
      label: 'API Calls (Monthly)',
      current: usage.apiCalls,
      limit: featureLimits.maxApiCalls,
      key: 'maxApiCalls' as const,
    },
  ];
  
  const currentPlan = subscription?.plan || 'free';
  
  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">Usage & Limits</CardTitle>
          <Badge variant="outline" className="capitalize">
            {currentPlan} Plan
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {limitItems.map((item) => {
          const percentage = getUsagePercentage(item.current, item.limit);
          const isUnlimited = item.limit === -1;
          const nearLimit = isNearLimit(item.current, item.limit);
          
          return (
            <div key={item.key} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <item.icon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {item.label}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  {nearLimit && !isUnlimited && (
                    <Badge variant="destructive" className="text-xs">
                      Near Limit
                    </Badge>
                  )}
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {item.current} / {isUnlimited ? (
                      <span className="flex items-center">
                        <Infinity className="h-3 w-3 ml-1" />
                      </span>
                    ) : (
                      formatLimit(item.limit)
                    )}
                  </span>
                </div>
              </div>
              
              {!isUnlimited && (
                <div className="space-y-1">
                  <Progress 
                    value={percentage} 
                    className="h-2"
                  />
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>{percentage.toFixed(0)}% used</span>
                    {nearLimit && (
                      <span className="text-red-500 dark:text-red-400">
                        Upgrade to increase limit
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
        
        {/* Data Refresh Rate */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ClockIcon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Data Refresh Rate
              </span>
            </div>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Every {featureLimits.dataRefreshRate}s
            </span>
          </div>
        </div>
        
        {/* Upgrade prompt for users near limits */}
        {limitItems.some(item => isNearLimit(item.current, item.limit)) && currentPlan !== 'premium' && (
          <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                    Approaching Limits
                  </p>
                  <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                    Upgrade your plan to get higher limits and more features.
                  </p>
                </div>
                <Button size="sm" variant="outline" className="ml-3">
                  Upgrade
                </Button>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default FeatureLimits;