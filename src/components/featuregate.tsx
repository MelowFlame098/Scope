import React from 'react';
import { useFeatureAccess, FeatureAccess } from '../hooks/useFeatureAccess';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import {
  LockClosedIcon,
  StarIcon,
  RocketLaunchIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { useSubscription } from '../contexts/SubscriptionContext';

interface FeatureGateProps {
  feature: keyof FeatureAccess;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  showUpgradePrompt?: boolean;
  className?: string;
}

const FeatureGate: React.FC<FeatureGateProps> = ({
  feature,
  children,
  fallback,
  showUpgradePrompt = true,
  className = '',
}) => {
  const { checkFeatureAccess, getUpgradeMessage, subscription } = useFeatureAccess();
  const { plans } = useSubscription();
  
  const hasAccess = checkFeatureAccess(feature);
  const upgradeMessage = getUpgradeMessage(feature);
  // Map backend plan format to frontend format
  const backendPlan = subscription?.plan?.toLowerCase() || 'free';
  const currentPlan = backendPlan === 'basic' ? 'basic' : backendPlan === 'premium' ? 'premium' : 'free';
  
  if (hasAccess) {
    return <div className={className}>{children}</div>;
  }
  
  if (fallback) {
    return <div className={className}>{fallback}</div>;
  }
  
  if (!showUpgradePrompt) {
    return null;
  }
  
  const getRecommendedPlan = () => {
    if (currentPlan === 'free') {
      return plans.find(p => p.id === 'basic') || plans[1];
    }
    return plans.find(p => p.id === 'premium') || plans[2];
  };
  
  const recommendedPlan = getRecommendedPlan();
  
  return (
    <div className={`${className} relative`}>
      <Card className="border-2 border-dashed border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800/50">
        <CardHeader className="text-center pb-4">
          <div className="flex items-center justify-center mb-2">
            <div className="p-3 bg-gray-200 dark:bg-gray-700 rounded-full">
              <LockClosedIcon className="h-8 w-8 text-gray-500 dark:text-gray-400" />
            </div>
          </div>
          <CardTitle className="text-lg font-semibold text-gray-700 dark:text-gray-300">
            Premium Feature
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          <Alert>
            <ExclamationTriangleIcon className="h-4 w-4" />
            <AlertDescription className="text-sm">
              {upgradeMessage}
            </AlertDescription>
          </Alert>
          
          {recommendedPlan && (
            <div className="space-y-3">
              <div className="p-4 bg-white dark:bg-gray-700 rounded-lg border">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {recommendedPlan.id === 'premium' ? (
                      <RocketLaunchIcon className="h-5 w-5 text-purple-500" />
                    ) : (
                      <StarIcon className="h-5 w-5 text-blue-500" />
                    )}
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {recommendedPlan.name}
                    </span>
                    {recommendedPlan.id === 'premium' && (
                      <Badge variant="secondary" className="bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                        Most Popular
                      </Badge>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      ${recommendedPlan.price}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      per month
                    </div>
                  </div>
                </div>
                
                <div className="space-y-1 mb-3">
                  {recommendedPlan.features.slice(0, 3).map((feature, index) => (
                    <div key={index} className="flex items-center text-sm text-gray-600 dark:text-gray-300">
                      <div className="w-1.5 h-1.5 bg-green-500 rounded-full mr-2"></div>
                      {feature}
                    </div>
                  ))}
                  {recommendedPlan.features.length > 3 && (
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      +{recommendedPlan.features.length - 3} more features
                    </div>
                  )}
                </div>
                
                <Button 
                  className="w-full" 
                  variant={recommendedPlan.id === 'premium' ? 'default' : 'outline'}
                  onClick={() => {
                    // This would typically open a subscription modal or navigate to pricing
                    console.log(`Upgrade to ${recommendedPlan.name}`);
                  }}
                >
                  Upgrade to {recommendedPlan.name}
                </Button>
              </div>
              
              <div className="text-xs text-gray-500 dark:text-gray-400">
                30-day free trial • Cancel anytime • No setup fees
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Blurred preview of the locked content */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="h-full w-full bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-lg flex items-center justify-center">
          <div className="opacity-20 scale-95 transform">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureGate;